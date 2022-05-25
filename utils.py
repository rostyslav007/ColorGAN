import cv2 as cv


def lab_tensor2cv_rgb(img_tensor, filename):
    nimg = (img_tensor.cpu().detach().permute(1, 2, 0).clamp(-1, 1).numpy() + 1) * 127.5
    ocvim = nimg.copy()
    fnam = filename
    cv.imwrite(fnam, ocvim)
