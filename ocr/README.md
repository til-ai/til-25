# OCR

## Input

Scanned document image provided as a 2481×3544 JPG files in grayscale.
![example document image](example.jpg)

## Output

Text contained within the document.

## Training Data

Provided with each image are also character/word/line/paragraph bounding boxes for use in training your OCR models. These are provided in `box` and `hocr` format, but they should be readily convertable to other formats as necessary.

### Box format

Character-level bounding boxes in `<symbol> <left> <bottom> <right> <top> <page>` format. See [Tesseract Docs](https://tesseract-ocr.github.io/tessdoc/tess3/Training-Tesseract-%E2%80%93-Make-Box-Files.html) for details.

### hOCR format

See the [Wikipedia article on the hOCR format](https://en.wikipedia.org/wiki/HOCR) for additional details.
