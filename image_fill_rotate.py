# Copyright (c) 2024 Jonathan S. Pollack (https://github.com/JPPhoto)
# Thanks to dwringer for help optimizing

import numpy as np
import numpy.typing as npt
from PIL import Image

from invokeai.app.invocations.baseinvocation import (
    BaseInvocation,
    InvocationContext,
    invocation,
)
from invokeai.app.invocations.fields import InputField, WithBoard, WithMetadata
from invokeai.app.invocations.primitives import ImageField, ImageOutput


@invocation("image_fill_rotate", title="Image Fill and Rotate", tags=["image_fill_rotate"], version="1.1.0")
class ImageFillRotateInvocation(BaseInvocation, WithBoard, WithMetadata):
    """Fills a rectangle by tiling and rotating an image"""

    image: ImageField = InputField(description="The image to add film grain to")
    angle: float = InputField(description="Angle to rotate the source image", default=0.0)
    width: int = InputField(gt=0, description="Target width", default=1024)
    height: int = InputField(gt=0, description="Target height", default=1024)

    def get_tiled_rotated_image(
        self, width: int, height: int, image: npt.NDArray[np.float64], angle: float
    ) -> npt.NDArray[np.float64]:
        new_img = np.empty(([height, width])) if len(image.shape) == 2 else np.empty(([height, width, image.shape[2]]))
        indices = np.indices((height, width))
        oheight, owidth = image.shape[0], image.shape[1]
        oheight2, owidth2 = oheight // 2, owidth // 2
        angle = np.pi * angle / 180
        sin, cos = np.sin(angle), np.cos(angle)

        xc = indices[1]  # adjust to x alone for offset
        yc = indices[0]  # adjust to y alone for offset for offset
        nx = xc * cos - yc * sin
        ny = xc * sin + yc * cos
        newx = nx + owidth2
        newy = ny + oheight2

        newx = np.where(
            owidth <= newx,
            newx - owidth * (newx / owidth).astype("int"),
            np.where(newx < 0, newx + owidth * (np.abs((newx / owidth).astype("int")) + 1), newx),
        )

        newy = np.where(
            oheight <= newy,
            newy - oheight * (newy / oheight).astype("int"),
            np.where(newy < 0, newy + oheight * (np.abs((newy / oheight).astype("int")) + 1), newy),
        )

        newx = (newx % owidth).astype("int")
        newy = (newy % oheight).astype("int")

        new_img[indices[0], indices[1], :] = image[newy, newx]
        return new_img

    def invoke(self, context: InvocationContext) -> ImageOutput:
        image = context.images.get_pil(self.image.image_name)

        image = np.array(image) / 255.0
        image = self.get_tiled_rotated_image(self.width, self.height, image, self.angle)
        image = Image.fromarray((image * 255.0).astype("uint8"))

        image_dto = context.images.save(image=image)

        return ImageOutput.build(image_dto)
