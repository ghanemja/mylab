"""Help text for imshow (overrides the older shared imagesc/image entry)."""

HELP = {
    "imshow": (["imshow(I)", "imshow(I, [low high])", "imshow(I, [])", "imshow(X, map)", "imshow(filename)",
                "h = imshow(...)"],
               "Shows an image with square pixels and no axis ticks. Grayscale images use the gray colormap over "
               "the range of their class (0 to 255 for uint8, 0 to 1 for double; black and white for logical); "
               "[low high] sets the range and [] uses the image's own minimum and maximum. M-by-N-by-3 arrays "
               "are shown as colour, X with a colormap as an indexed image, and a file name is read first."),
}
