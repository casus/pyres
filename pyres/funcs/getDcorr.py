import numpy as np
from scipy.fft import fftshift, fftn
import matplotlib.pyplot as plt
from .getCorrcoef import getCorrcoef
from .getDcorrLocalMax import getDcorrLocalMax

# from funcs.loadData import load_data
from .getRadAvg import getRadAvg
from .linmap import linmap

Nr = 50
Ng = 10
r = np.linspace(0, 1, Nr)


def getDcorr(im, r=None, Ng=None, figID=None):
    if r is None:
        r = np.linspace(0, 1, 50)
    if Ng is None:
        Ng = 10

    if isinstance(figID, str) or figID > 1:
        figID = 0
        fastMode = 1
    else:
        fastMode = 0

    if len(r) < 30:
        r = np.linspace(np.min(r), np.max(r), 30)
    if Ng < 5:
        Ng = 5

    im = im.astype(np.float64)  # Convert 'im' to single precision (Check for float64)

    # Ensure that 'im' has an odd number of pixels in both dimensions
    if im.shape[0] % 2 == 0:
        im = im[:-1, :]
    if im.shape[1] % 2 == 0:
        im = im[:, :-1]
    print(f"Image shape is : {im.shape}")

    X, Y = np.meshgrid(np.linspace(-1, 1, im.shape[1]), np.linspace(-1, 1, im.shape[0]))
    R = np.sqrt(X ** 2 + Y ** 2)
    Nr = len(r)
    #print(f"Nr is : {Nr}")

    # Compute the Fourier normalized image 'In'
    In = fftshift(fftn(fftshift(im)))
    In = In / np.abs(In)
    In[np.isinf(In)] = 0
    In[np.isnan(In)] = 0

    # Create the binary mask 'mask0' based on the radial distance 'R'
    mask0 = R ** 2 < 1 ** 2

    # Restrict the analysis to the region where 'r < 1'
    In = mask0 * In

    # plt.imshow(np.abs(In), cmap="gray") 
    # plt.title("Fourier Normalized Image")
    # plt.colorbar()
    # plt.show()

    if figID is not None and figID > 0:
        print("Computing dcorr:")

    # Compute the Fourier transform 'Ik' of 'im'
    Ik = mask0 * fftshift(fftn(fftshift(im)))
    #print(f"Fourier transform is Ik: {Ik}")

    #print(f"Fourier transform shape is : {Ik.shape}")  # 255, 255
    # plt.imshow(np.abs(Ik), cmap="gray") 
    # plt.title("Fourier Transform")
    # plt.colorbar()
    # plt.show()

    # Compute the amplitude 'c' of the Fourier transform
    c = np.sqrt(np.sum(np.abs(Ik) ** 2))
    #print(f"Amplitude of the Fourier transform is : {c:.3f}")

    # Create a linearly spaced vector 'r0' 
    r0 = np.linspace(r[0], r[-1], Nr)
    #print(f"Linear spacing of the Fourier transform is : {r0}")
    #print(f"r0 shape is : {r0.shape}")
    # Initialize 'd0' as an empty array
    d0 = np.zeros(Nr)
    # d0 = []
    # Iterate through 'r0' in reverse order
    for k in range(len(r0) - 1, -1, -1):
        # Calculate the correlation coefficient 'cc'
        cc = getCorrcoef(Ik, (R ** 2 < r0[k] ** 2) * In, c)

        # Check if 'cc' is NaN and replace with 0 if necessary
        if np.isnan(cc):
            cc = 0

        # Append 'cc' to 'd0'
        d0[k] = cc

        # If in 'fastMode', look for local maxima and update 'ind0' and 'snr0'
        if fastMode == 1:
            ind0, snr0 = getDcorrLocalMax(d0[k:])
            ind0 += k
            if ind0 > k:
                #print("Break the loop with ind0 > k")
                break
            
    if fastMode == 0:
        ind0, _ = getDcorrLocalMax(d0[k:])
        snr0 = d0[ind0]

    k0 = r[ind0]
    #print("k0 is : ", k0)
    gMax = 2 / r0[ind0]
    #print(f"gMax is : {gMax}")
    if np.isinf(gMax):
        gMax = max(im.shape[0], im.shape[1]) / 2
        #print(f"gMax is : {gMax}")

    # search of highest frequency peak
    #print(im.shape[0] / 4)
    #print(np.exp(np.linspace(np.log(gMax), np.log(0.15), Ng)))
    # g = np.concatenate(([[im.shape[0] / 4]], np.exp(np.linspace(np.log(gMax), np.log(0.15), Ng))))
    g = [im.shape[0] / 4] + list(np.exp(np.linspace(np.log(gMax), np.log(0.15), Ng)))
    #print(f"g is :{g}")

    #d = np.zeros((Nr, 2 * Ng))
    d = np.zeros((Nr, 2 * Ng + 1))
    #print(f"d shape is : {d.shape}")
    # Assuming Ng and Nr are defined and have integer values
    kc = [0] * (2 * Ng * Nr)  # This initializes a list with enough zeros
    #print(f"kc  is : {kc}")
    SNR = [0] * (2 * Ng * Nr)  # Same for SNR
    #print(f"SNR  is : {SNR}")


    if fastMode == 0:
        ind0 = 1
    elif ind0 > 1:
        ind0 = ind0 - 1

    # Two-step refinement
    for refin in range(1, 3):
        for h in range(len(g)):
            Ir = Ik * (1 - np.exp(-2 * g[h] * g[h] * R ** 2))
            c = np.sqrt(np.sum(np.abs(Ir) ** 2))

            for k in range(len(r) - 1, ind0 - 1, -1):
                #print(" the value of k is : ", k)
                mask = R ** 2 < r[k] ** 2
                cc = getCorrcoef(Ir[mask], In[mask], c)
                cc = 0 if np.isnan(cc) else cc
                #print("refine CC: ", cc)
                #print(k, h + Ng * (refin - 1))
                d[k, h + Ng * (refin - 1)] = cc

                if fastMode:
                    ind, snr = getDcorrLocalMax(d[k:, h + Ng * (refin - 1)])
                    #print("index ind is : ", ind)
                    ind += k
                    if ind > k:
                        break

            if not fastMode:
                ind = getDcorrLocalMax(d[k:, h + Ng * (refin - 1)])
                snr = d[ind, h + Ng * (refin - 1)]
            kc[h + Ng * (refin - 1)] = r[ind]
            SNR[h + Ng * (refin - 1)] = snr
            if figID is not None and figID > 0:
                print("-", end="")

    # Refine high-pass threshold and radius sampling
    if refin == 1:
        indmax = np.where(kc == np.max(kc))[0]
        ind1 = indmax[-1]
        g1, g2 = (
            (im.shape[0], g[0])
            if ind1 == 0
            else (g[ind1 - 1], g[min(ind1, len(g) - 1)])
        )
        g = np.exp(np.linspace(np.log(g1), np.log(g2), Ng))
        r1 = kc[indmax[-1]] - (r[1] - r[0])
        r2 = kc[indmax[-1]] + 0.4
        r = np.linspace(max(r1, 0), min(r2, 1), Nr)
        ind0 = 0

    if figID is not None and figID > 0:
        print(" -- Computation done-- ")

    kc.append(k0)  # Assuming kc is already a list
    SNR.append(snr0)
    #print(f"kc is : {kc}")
    #print(f"SNR is : {SNR}")

    kc = np.array(kc)  # Convert lists to numpy arrays for element-wise operations
    SNR = np.array(SNR)
    kc[SNR < 0.05] = 0
    SNR[SNR < 0.05] = 0
    

    if kc.size > 0:
        kcMax = np.max(kc)
        ind = np.argmax(kc)
        AMax = SNR[ind]
        A0 = snr0  # average image contrast has to be estimated from original image
    else:
        kcMax = r[1]
        Amax = 0
        res = r[1]
        A0 = 0

    # if figID is not None and figID > 0:
    #     print(f"getRadAvg = {np.log(np.abs(Ik) + 1)}")
    #     radAv = getRadAvg(np.log(np.abs(Ik) + 1))
    #     lnwd = 1.5
    #     plt.figure(figID)
    #     plt.plot(r0, d[:, :Ng], color=(0.2, 0.2, 0.2, 0.5))
    #     radAv[0] = radAv[1]
    #     radAv[-1] = radAv[-2]
    #     plt.plot(
    #         np.linspace(0, 1, len(radAv)),
    #         linmap(radAv, 0, 1),
    #         linewidth=lnwd,
    #         color=(1, 0, 1),
    #     )
    #     for n in range(Ng):
    #         plt.plot(r2, d[:, n + Ng], color=(0, 0, n / Ng))
    #     plt.plot(r0, d0, linewidth=lnwd, color="g")
    #     plt.plot([kcMax, kcMax], [0, 1], "k")
    #     for k in range(len(kc)):
    #         plt.plot(kc[k], SNR[k], "bx", linewidth=1)

    #     plt.title(f"Dcor analysis: res ~ {kcMax:.4f}, SNR ~ {A0:.4f}")
    #     plt.xlim([0, 1])
    #     plt.ylim([0, 1])
    #     plt.xlabel("Normalized spatial frequencies")
    #     plt.ylabel("C.c. coefficients")
    #     plt.show()  # Display the plot
    return kcMax, A0, d0, d
