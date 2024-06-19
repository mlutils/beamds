echo "Installing tesseract-ocr"

apt install -y tesseract-ocr-eng
apt install -y tesseract-ocr-osd
apt install -y libtesseract-dev
apt install -y libarchive-dev libleptonica-dev libtesseract-dev
apt install -y libarchive13
apt install -y tesseract-ocr
apt install -y tesseract-ocr-ara
apt install -y libtesseract5

echo "Installing pytesseract client packages"
pip install pytesseract tesserocr