import imageio
from matplotlib.pyplot import imread

ims = []
ims.append(imread("spectra/Young_Old/Thresh_400/spectra_img.png"))
ims.append(imread("spectra/Young_Old/Thresh_800/spectra_img.png"))
ims.append(imread("spectra/Young_Old/Thresh_1200/spectra_img.png"))
ims.append(imread("spectra/Young_Old/Thresh_1600/spectra_img.png"))
ims.append(imread("spectra/Young_Old/Thresh_2000/spectra_img.png"))
imageio.mimsave('spectra/Young_Old/animated.gif', ims, duration = 1)

ims = []
ims.append(imread("spectra/Coma/Parcel_90/Thresh_400/spectra_img.png"))
ims.append(imread("spectra/Coma/Parcel_90/Thresh_800/spectra_img.png"))
ims.append(imread("spectra/Coma/Parcel_90/Thresh_1200/spectra_img.png"))
ims.append(imread("spectra/Coma/Parcel_90/Thresh_1600/spectra_img.png"))
ims.append(imread("spectra/Coma/Parcel_90/Thresh_2000/spectra_img.png"))
imageio.mimsave('spectra/Coma/Parcel_90/animated.gif', ims, duration = 1)

ims = []
ims.append(imread("spectra/Coma/Parcel_417/Thresh_8663/spectra_img.png"))
ims.append(imread("spectra/Coma/Parcel_417/Thresh_17326/spectra_img.png"))
ims.append(imread("spectra/Coma/Parcel_417/Thresh_25988/spectra_img.png"))
ims.append(imread("spectra/Coma/Parcel_417/Thresh_34651/spectra_img.png"))
ims.append(imread("spectra/Coma/Parcel_417/Thresh_43314/spectra_img.png"))
imageio.mimsave('spectra/Coma/Parcel_417/animated.gif', ims, duration = 1)


ims = []
ims.append(imread("spectra/HCP/Parcel_90/Thresh_400/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_90/Thresh_800/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_90/Thresh_1200/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_90/Thresh_1600/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_90/Thresh_2000/spectra_img.png"))
imageio.mimsave('spectra/HCP/Parcel_90/animated.gif', ims, duration = 1)


ims = []
ims.append(imread("spectra/HCP/Parcel_384/Thresh_7344/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_384/Thresh_14689/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_384/Thresh_22033/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_384/Thresh_29378/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_384/Thresh_36722/spectra_img.png"))
imageio.mimsave('spectra/HCP/Parcel_384/animated.gif', ims, duration = 1)


ims = []
ims.append(imread("spectra/HCP/Parcel_459/Thresh_10498/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_459/Thresh_20996/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_459/Thresh_31494/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_459/Thresh_41992/spectra_img.png"))
ims.append(imread("spectra/HCP/Parcel_459/Thresh_52490/spectra_img.png"))
imageio.mimsave('spectra/HCP/Parcel_459/animated.gif', ims, duration = 1)
