import io
import os
import torch
import numpy as np
import boto3
from PIL import Image, ImageSequence, ImageOps, ImageFile, UnidentifiedImageError
from botocore.exceptions import NoCredentialsError


def pillow(fn, arg):
    prev_value = None
    try:
        x = fn(arg)
    except (
        OSError,
        UnidentifiedImageError,
        ValueError,
    ):  # PIL issues #4472 and #2445, also fixes ComfyUI issue #3416
        prev_value = ImageFile.LOAD_TRUNCATED_IMAGES
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        x = fn(arg)
    finally:
        if prev_value is not None:
            ImageFile.LOAD_TRUNCATED_IMAGES = prev_value
    return x


def s3_upload_file(client, bucket, key, local_path):
    try:
        client.upload_file(local_path, bucket, key)
        return key
    except NoCredentialsError:
        print("Credentials not available or not valid.")
    except Exception as e:
        print(f"Failed to upload file to S3: {e}")


def direct_s3_save_file(client, bucket, key, buff):
    buff.seek(0)
    return client.put_object(Body=buff, Key=key, Bucket=bucket)


def direct_s3_load_file(client, bucket, key):
    outfile = io.BytesIO()
    client.download_fileobj(bucket, key, outfile)
    outfile.seek(0)
    return outfile


def direct_s3_init_client(region="", ak="", sk=""):
    client = None
    region = os.getenv("AWS_DEFAULT_REGION", region)
    ak = os.getenv("AWS_ACCESS_KEY_ID", ak)
    sk = os.getenv("AWS_SECRET_ACCESS_KEY", sk)
    if region == "" and ak == "" and sk == "":
        client = boto3.client("s3")
    elif ak == "" and sk == "":
        client = boto3.client("s3", region_name=region)
    elif ak != "" and sk != "":
        client = boto3.client(
            "s3", region_name=region, aws_access_key_id=ak, aws_secret_access_key=sk
        )
    else:
        client = boto3.client("s3")
    return client


def remove_extensions(pathname):
    extensions = [".jpg", ".png", ".webp"]
    for ext in extensions:
        if pathname.endswith(ext):
            return pathname[: -len(ext)]
    return pathname


class DirectSaveImageToS3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                "pathname": (
                    "STRING",
                    {"multiline": False, "default": "pathname for file"},
                ),
            },
            "optional": {
                "region": ("STRING", {"multiline": False, "default": ""}),
                "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                "aws_sk": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("FILES",)
    RETURN_NAMES = ("files",)
    FUNCTION = "save_image_to_s3"
    CATEGORY = "DirectS3"
    OUTPUT_NODE = True

    def save_image_to_s3(self, images, region, aws_ak, aws_sk, s3_bucket, pathname):
        client = direct_s3_init_client(region, aws_ak, aws_sk)
        results = list()
        for batch_number, image in enumerate(images):
            i = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")

            prefix = remove_extensions(pathname)
            filename = (
                "%s_%i.png" % (prefix, batch_number)
                if len(images) > 1
                else "%s.png" % (prefix)
            )

            direct_s3_save_file(client, s3_bucket, filename, img_byte_arr)

            results.append({
                "filename": filename,
                "subfolder": "",
                "type": "output",
                "bucket": s3_bucket,
                "pathname": pathname,
            })
        return {"ui": {"images": results}, "result": (results)}


class DirectLoadImageFromS3:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                "pathname": (
                    "STRING",
                    {"multiline": False, "default": "pathname for file"},
                ),
            },
            "optional": {
                "region": ("STRING", {"multiline": False, "default": ""}),
                "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                "aws_sk": ("STRING", {"multiline": False, "default": ""}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image_from_s3"
    CATEGORY = "DirectS3"

    def load_image_from_s3(self, region, aws_ak, aws_sk, s3_bucket, pathname):
        client = direct_s3_init_client(region, aws_ak, aws_sk)
        img = pillow(Image.open, direct_s3_load_file(client, s3_bucket, pathname))

        output_images = []
        output_masks = []
        w, h = None, None

        excluded_formats = ["MPO"]

        for i in ImageSequence.Iterator(img):
            i = pillow(ImageOps.exif_transpose, i)

            if i is None:
                continue
            if i.mode == "I":
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if "A" in i.getbands():
                mask = np.array(i.getchannel("A")).astype(np.float32) / 255.0
                mask = 1.0 - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)


class SaveVHSVideoFilesS3:
    def __init__(self):
        self.s3_output_dir = os.getenv("S3_OUTPUT_DIR")
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filenames": ("VHS_FILENAMES",),
                "s3_bucket": ("STRING", {"multiline": False, "default": "s3_bucket"}),
                "s3_pathname": ("STRING", {"multiline": False, "default": ""}),
                "s3_file_prefix": ("STRING", {"multiline": False, "default": ""}),
            },
            "optional": {
                "region": ("STRING", {"multiline": False, "default": ""}),
                "aws_ak": ("STRING", {"multiline": False, "default": ""}),
                "aws_sk": ("STRING", {"multiline": False, "default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("FILES",)
    RETURN_NAMES = ("files",)
    FUNCTION = "save_video_files"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = "DirectS3"

    def save_video_files(
        self, filenames, s3_bucket, s3_pathname, s3_file_prefix, region, aws_ak, aws_sk
    ):
        client = direct_s3_init_client(region, aws_ak, aws_sk)
        results = list()
        tf, filenames = filenames

        for batch_number, filename in enumerate(filenames):
            localpath = filename
            filename = filename.split("/")[-1]
            s3_filename = f"{s3_file_prefix}{filename}"
            s3_full_filename = (
                f"{s3_filename}"
                if s3_pathname == ""
                else f"{s3_pathname}/{s3_filename}"
            )

            s3_upload_file(client, s3_bucket, s3_full_filename, localpath)

            results.append({
                "filename": s3_full_filename,
                "subfolder": "",
                "type": "output",
                "bucket": s3_bucket,
                "pathname": s3_pathname,
            })
        return {"ui": {"images": results}, "result": (results)}


NODE_CLASS_MAPPINGS = {
    "Direct Save Image To S3": DirectSaveImageToS3,
    "Direct Load Image From S3": DirectLoadImageFromS3,
    "Save VHS Video to S3": SaveVHSVideoFilesS3,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CKM Direct Save Image To S3": "Direct Save Your Image to S3",
    "CKM Direct Load Image From S3": "Direct Load Your Image From S3",
}

