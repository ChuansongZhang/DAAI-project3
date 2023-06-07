from PIL import Image


def load_img(x_path, y_path=None, cv2=False, x_hpf_path=None):

  x = Image.open(x_path)
  y = Image.open(y_path) if y_path is not None else None
  x_hpf = Image.open(x_hpf_path) if x_hpf_path is not None else None

  return x, y, x_hpf
