import numpy as np
import matplotlib.pyplot as plt

def getDcorr1D(sig, r=None, Ng=10, figID=0):
    if figID == 0:
        fastMode = False
    else:
        fastMode = True

    if r is None:
        r = np.linspace(0, 1, 50)

    if len(r) < 30:
        r = np.linspace(min(r), max(r), min(30, (len(sig) + 1) // 2))
    elif len(r) > (len(sig) + 1) // 2:
        r = np.linspace(min(r), max(r), (len(sig) + 1) // 2)

    if Ng < 5:
        Ng = 5

    if sig.shape[0] > 1 and sig.shape[1] == 1:
        sig = sig.T

    sig = sig[:-(len(sig) % 2)]  # odd number of pixels
    R = np.abs(np.linspace(-1, 1, len(sig)))
    Nr = len(r)

    Sn = np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig)))
    Sn /= np.abs(Sn)
    Sn[np.isinf(Sn)] = 0
    Sn[np.isnan(Sn)] = 0

    mask0 = R <= 1
    Sn = mask0 * Sn  # restrict all the analysis to the region r < 1

    if figID:
        print('Computing dcorr: ')

    Sk = mask0 * np.fft.fftshift(np.fft.fft(np.fft.fftshift(sig)))
    c = np.sqrt(np.sum(np.sum(Sk * np.conj(Sk))))

    r0 = np.linspace(r[0], r[-1], Nr)
    d0 = np.zeros(len(r0))

    for k in range(len(r0) - 1, -1, -1):
        cc = getCorrcoef(Sk, (R < r0[k]) * Sn, c)
        if np.isnan(cc):
            cc = 0
        d0[k] = cc

        if fastMode:
            ind0, snr0 = getDcorrLocalMax(d0[k:])
            ind0 += k
            if ind0 > k:  # found a local maximum, skip the calculation
                break

    if fastMode == 0:
        ind0 = getDcorrLocalMax(d0[k:])
        snr0 = d0[ind0]

    k0 = r[ind0]

    gMax = 2 / r0[ind0]
    if np.isinf(gMax):
        gMax = max(len(sig)) / 2

    g = np.concatenate(([len(sig) / 4 / 4], np.exp(np.linspace(np.log(gMax), np.log(0.15), Ng))))
    d = np.zeros((Nr, 2 * Ng))
    kc = np.zeros(2 * Ng + 1)
    SNR = np.zeros(2 * Ng + 1)

    if fastMode == 0:
        ind0 = 1
    else:
        if ind0 > 1:
            ind0 -= 1

    for refin in range(2):
        for h in range(len(g)):
            Sr = Sk * (1 - np.exp(-2 * g[h] * g[h] * R**2))  # Fourier Gaussian filtering
            c = np.sqrt(np.sum(np.sum(np.abs(Sr)**2)))

            for k in range(len(r) - 1, ind0 - 1, -1):
                mask = R < r[k]
                cc = getCorrcoef(Sr[mask], Sn[mask], c)
                if np.isnan(cc):
                    cc = 0
                d[k, h + Ng * refin] = cc

                if fastMode:
                    ind, snr = getDcorrLocalMax(d[k:, h + Ng * refin])
                    ind += k
                    if ind > k:  # found a local maximum, skip the calculation
                        break

            if fastMode == 0:
                ind = getDcorrLocalMax(d[k:, h + Ng * refin])
                snr = d[ind, h + Ng * refin]
                ind += k

            kc[h + Ng * refin + 1] = r[ind]
            SNR[h + Ng * refin + 1] = snr

            if figID:
                print('-', end='')

    if refin == 0:
        indmax = np.argmax(kc)
        ind1 = indmax
        if ind1 == 1:  # peak only without highpass
            ind1 = 1
            ind2 = 2
            g1 = len(sig)
            g2 = g[0]
        elif ind1 >= len(g):
            ind2 = ind1 - 1
            ind1 = ind1 - 2
            g1 = g[ind1]
            g2 = g[ind2]
        else:
            ind2 = ind1
            ind1 = ind1 - 1
            g1 = g[ind1]
            g2 = g[ind2]

        g = np.exp(np.linspace(np.log(g1), np.log(g2), Ng))

        r1 = kc[indmax] - (r[2] - r[1])
        r2 = kc[indmax] + 0.3
        if r1 < 0:
            r1 = 0
        if r2 > 1:
            r2 = 1
        r = np.linspace(r1, min(r2, r[-1]), Nr)
        ind0 = 1
        r2 = r

    if figID:
        print('-- Computation done --')

    kc = np.append(kc, k0)
    SNR = np.append(SNR, snr0)

    kc[SNR < 0.05] = 0
    SNR[SNR < 0.05] = 0

    snr = SNR

    if len(kc) != 0:
        kcMax = np.max(kc)
        ind = np.argmax(kc)
        AMax = SNR[ind]
        A0 = snr0  # average image contrast has to be estimated from original image
    else:
        kcMax = r[1]
        Amax = 0
        res = r[1]
        A0 = 0

    if figID:
        radAv = np.log(np.abs(gather(Sk[(len(sig) + 1) // 2:])))
        lnwd = 1.5
        plt.plot(r0, d[:, :Ng], color=[0.2, 0.2, 0.2, 0.5])
        radAv[0] = radAv[1]  # for plot
        radAv[-1] = radAv[-2]
        plt.plot(np.linspace(0, 1, len(radAv)), linmap(radAv, 0, 1), linewidth=lnwd, color=[1, 0, 1])
        for n in range(Ng):
            plt.plot(r2, d[:, n + Ng], color=[0, 0, (n - 1) / Ng])
        plt.plot(r0, d0, linewidth=lnwd, color='g')
        plt.plot([kcMax, kcMax], [0, 1], 'k')
        for k in range(len(kc)):
            plt.plot(kc[k], snr[k], 'bx', linewidth=1)
        plt.title(f"Dcor analysis : res ~ {kcMax:.4f}, SNR ~ {A0:.4f}")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel("Normalized spatial frequencies")
        plt.ylabel("C.c. coefficients")
        plt.show()
