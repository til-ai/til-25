# OCR

TODO! update example jpg

## Input

Scanned document image provided as a 2481×3544 grayscale JPG files.
![example document image](example.jpg)

## Output

Text contained within the document.

## Training Data

Provided with each image are also word/line/paragraph bounding boxes for use in training your OCR models. These are provided in `hocr` format, but they should be readily convertable to other formats as necessary.

### hOCR format

See the [Wikipedia article on the hOCR format](https://en.wikipedia.org/wiki/HOCR) for additional details.
