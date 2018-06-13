"""

"""

import numpy as np
from scipy.signal import convolve2d


def attenuation_border_matrix(nz, nx, pad=20, factor=0.015):
    """A matrix used to avoid reflection on borders.

    Parameters
    ----------
    nz, nx : int
        Number of samples in vertical and horizontal directions. Note that
        the expected dimensions are of the non-padded matrix.
    pad : int
        The padding size.
    
    Returns
    -------
    array_like
        A matrix (2D array) that sould be multiplied by the wave_field of each
        iteration of the numerical solution to avoid reflection on borders.
    
    Notes
    -----
    This function will output a matrix that has `pad` more lines and `2*pad` more
    columns than the given dimensions. It should be used with matrices that were
    accordingly padded.

    For more technical details see https://library.seg.org/doi/abs/10.1190/1.1441945

    """
    coef = np.exp(-(factor*np.arange(pad))**2.0)

    abm = np.ones((nz+pad, nx+2*pad), dtype=float)
    
    abm[-pad:, :] *= coef.reshape((-1, 1))
    abm[:, :pad] *= coef[::-1]
    abm[:, -pad:] *= coef

    return abm


def simulate_2d_wave_equation(velocity, dz, dx, dt, nt, sources, sz, sx, pad=20, factor=0.015, skip=1):
    """Calculate 2D-wave-field given velocity field and source.

    Parameters
    ----------
    velocity : array_like
        Velocity (m/s) on each position of the grid.
    dz, dx : float
        Grid spacing (meters) in vertical and horizontal directions.
    dt : float
        Time interval (seconds) for the output.
    nt : int
        Number of time frames to be calculated.
    sources : list of array_like
        1D-arrays defining the sources. All arrays must have length `nt`.
    sz, sx : list of int
        Horizontal and vertical indices of the sources. Note that sources
        must be positioned in a grid node.
    pad : int
        Number of columns/lines that will be padded to effectively solve
        the wave equation in order to avoid border reflections.
    skip : int, optional
        Number of time frames that will be skipped before saving the next
        one.
    
    Returns
    -------
    wave_field : array_like
        3D-array with the wavefield. The first dimension is time, and
        the second is depth, so that `wave_field[i]` is the i-esimal
        time frame and `wave_field[:, 0, :]` is the seismogram.
    
    Notes
    -----
    The solution for the wave equation is calculated using 4th order
    approximation for spatial derivatives and 2nd order for time
    derivative. The user must be sure that the parameters will satisfy
    stability and numerical dispersion conditions.

    """
    total_pad = pad + 2
    sim_v = velocity.copy()**2*dt**2

    sim_v = np.append(np.tile(sim_v[:, 0], (total_pad, 1)).T, sim_v, axis=1)
    sim_v = np.append(sim_v, np.tile(sim_v[:, -1], (total_pad, 1)).T, axis=1)
    sim_v = np.append(sim_v, np.tile(sim_v[-1, :], (total_pad, 1)), axis=0)

    abm = np.empty_like(sim_v)
    abm[:-2, 2:-2] = attenuation_border_matrix(*velocity.shape, pad, factor)
    abm[-2:, :] = 0.0
    abm[:, :2] = 0.0
    abm[:, -2:] = 0.0

    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.imshow(velocity)
    # plt.subplot(1, 3, 2)
    # plt.imshow(sim_v)
    # plt.subplot(1, 3, 3)
    # plt.imshow(abm)
    # plt.show()

    f1z = 1.0/12.0/dz**2
    f2z = 4.0/3.0/dz**2
    f1x = 1.0/12.0/dx**2
    f2x = 4.0/3.0/dx**2
    ft = 2.5/dx**2 + 2.5/dz**2
    
    pseudo_wave_field = np.zeros((3,) + sim_v.shape, dtype=float)
    # wave_field = np.empty((nt//skip,) + velocity.shape)
    wave_field = np.empty((nt//skip,) + sim_v.shape)

    # operator = np.zeros((5, 5), dtype=float)
    # operator[0, 2] = -f1z
    # operator[1, 2] = f2z
    # operator[2, 2] = ft
    # operator[3, 2] = f2z
    # operator[4, 2] = -f1z
    # operator[2, 0] = -f1x
    # operator[2, 1] = f2x
    # operator[2, 3] = f2x
    # operator[2, 4] = -f1x

    for i in range(nt):
        i0 = (i + 2) % 3
        i1 = (i + 1) % 3
        i2 = (i + 0) % 3

        pseudo_wave_field[i0] = -pseudo_wave_field[i1]*ft

        pseudo_wave_field[i0, :-2, :] -= f1z*pseudo_wave_field[i1, 2:, :]
        pseudo_wave_field[i0, :-1, :] += f2z*pseudo_wave_field[i1, 1:, :]
        pseudo_wave_field[i0, 1:, :] += f2z*pseudo_wave_field[i1, :-1, :]
        pseudo_wave_field[i0, 2:, :] -= f1z*pseudo_wave_field[i1, :-2, :]
        
        pseudo_wave_field[i0, :, :-2] -= f1x*pseudo_wave_field[i1, :, 2:]
        pseudo_wave_field[i0, :, :-1] += f2x*pseudo_wave_field[i1, :, 1:]
        pseudo_wave_field[i0, :, 1:] += f2x*pseudo_wave_field[i1, :, :-1]
        pseudo_wave_field[i0, :, 2:] -= f1x*pseudo_wave_field[i1, :, :-2]

        # pseudo_wave_field[i0] = convolve2d(pseudo_wave_field[i1], operator, mode="same")

        for source, sz_, sx_ in zip(sources, sz, sx):
            pseudo_wave_field[i0, sz_, sx_+total_pad] += source[i]

        pseudo_wave_field[i0] *= sim_v

        pseudo_wave_field[i0] += 2.0*pseudo_wave_field[i1] - pseudo_wave_field[i2]

        # if (710 <= i <= 755) and (i % 3 == 0):
        #     plt.figure()
        #     plt.subplot(1, 3, 1)
        #     plt.imshow(pseudo_wave_field[i0, 27:58, 0:31], vmin=-0.0005, vmax=0.0005, interpolation="none")
        #     plt.subplot(1, 3, 2)
        #     plt.imshow(abm[27:58, 0:31], vmin=0, vmax=1.0, interpolation="none")
        #     plt.subplot(1, 3, 3)
        #     plt.imshow(pseudo_wave_field[i0, 27:58, 0:31]*abm[27:58, 0:31], vmin=-0.0005, vmax=0.0005, interpolation="none")

        pseudo_wave_field[i0] *= abm

        if i % skip == 0:
            print("{}%".format(100.0*i/nt))
            # wave_field[i//skip] = pseudo_wave_field[i0, :-total_pad, total_pad:-total_pad]
            wave_field[i//skip] = pseudo_wave_field[i0]
    print("100.0%")
    # plt.show()
    return wave_field

def simulate_1d_wave_equation(velocity, dx, dt, nt, sources, sx, skip=1):
    """Calculate 2D-wave-field given velocity field and source.

    Parameters
    ----------
    velocity : array_like
        Velocity (m/s) on each position of the grid.
    dx : float
        Grid spacing (meters).
    dt : float
        Time interval (seconds) for the output.
    nt : int
        Number of time frames to be calculated.
    sources : list of array_like
        1D-arrays defining the sources. All arrays must have length `nt`.
    sx : list of int
        Indices of the sources. Note that sources must be positioned in a
        grid node.
    skip : int, optional
        Number of time frames that will be skipped before saving the next
        one.
    
    Returns
    -------
    wave_field : array_like
        2D-array with the wavefield. The first dimension is time, and
        the second is x, so that `wave_field[i]` is the i-esimal time
        frame and `wave_field[:, 0]` is the seismogram.
    
    Notes
    -----
    The solution for the wave equation is calculated using 4th order
    approximation for spatial derivatives and 2nd order for time
    derivative. The user must be sure that the parameters will satisfy
    stability and numerical dispersion conditions.

    """
    total_pad = 2
    sim_v = velocity.copy()**2*dt**2

    sim_v = np.append(np.repeat(sim_v[0], total_pad), sim_v)
    sim_v = np.append(sim_v, np.tile(sim_v[-1], total_pad))

    # TODO: implement 1D-ABM
    # abm = np.empty_like(sim_v)
    # abm[:-2, 2:-2] = attenuation_border_matrix(*velocity.shape, pad, factor)
    # abm[-2:, :] = 0.0
    # abm[:, :2] = 0.0
    # abm[:, -2:] = 0.0

    f1x = 1.0/12.0/dx**2
    f2x = 4.0/3.0/dx**2
    ft = 2.5/dx**2

    pseudo_wave_field = np.zeros((3,) + sim_v.shape, dtype=float)
    wave_field = np.empty((nt//skip,) + velocity.shape)
    # wave_field = np.empty((nt//skip,) + sim_v.shape)

    operator = np.array([-f1x, f2x, -ft, f2x, -f1x])

    for i in range(nt):
        i0 = (i + 2) % 3
        i1 = (i + 1) % 3
        i2 = (i + 0) % 3

        # pseudo_wave_field[i0] = -pseudo_wave_field[i1]*ft
        
        # pseudo_wave_field[i0, :-2] -= f1x*pseudo_wave_field[i1, 2:]
        # pseudo_wave_field[i0, :-1] += f2x*pseudo_wave_field[i1, 1:]
        # pseudo_wave_field[i0, 1:] += f2x*pseudo_wave_field[i1, :-1]
        # pseudo_wave_field[i0, 2:] -= f1x*pseudo_wave_field[i1, :-2]

        pseudo_wave_field[i0] = np.convolve(pseudo_wave_field[i1], operator, mode="same")

        for source, sx_ in zip(sources, sx):
            pseudo_wave_field[i0, sx_+total_pad] += source[i]

        pseudo_wave_field[i0] *= sim_v

        pseudo_wave_field[i0] += 2.0*pseudo_wave_field[i1] - pseudo_wave_field[i2]

        if i % skip == 0:
            print("{}%".format(100.0*i/nt))
            wave_field[i//skip] = pseudo_wave_field[i0, total_pad:-total_pad]
            # wave_field[i//skip] = pseudo_wave_field[i0]
    print("100.0%")
    # plt.show()
    return wave_field

if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt
    # plt.rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick-7.0.7-Q16\magick.exe'
    import matplotlib.animation as manimation
    import time

    dt = 1.0/2400.0
    dz = 6.25
    dx = 6.25

    v = np.ones((401, 401), dtype=float)*2250.0
    v[int((v.shape[0]-1)//2):] += 500.0

    nt = 4000
    skip = 40

    ts = np.arange(-128, 129)*dt
    freq = 20.0
    s = np.zeros(nt, dtype=float)
    s[:len(ts)] = (1.0 - 2.0*(np.pi*freq*ts)**2.0)*np.exp(-(np.pi*freq*ts)**2.0)

    from scipy import fftpack

    s_f = fftpack.rfft(s[:len(ts)])
    f = fftpack.rfftfreq(len(ts), dt)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(ts, s[:len(ts)])
    plt.subplot(2, 1, 2)
    plt.plot(f, s_f**2)
    plt.show()
    
    sources = [s]
    sz = [0]
    sx = [int((v.shape[0]-1)//2)]

    pad = 20
    factor = 0.015

    t0 = time.clock()
    wave_field = simulate_2d_wave_equation(v, dz, dx, dt, nt, sources, sz, sx, pad, factor, skip)
    t1 = time.clock()
    print(t1-t0)

    perc = 1
    clip = np.percentile(np.abs(wave_field.flatten()), 100-perc)

    fig = plt.figure()

    plt.subplot(1, 2, 1)
    plt.xlim(0, wave_field.shape[2]-1)
    plt.ylim(0, wave_field.shape[1]-1)
    im1 = plt.imshow(wave_field[0, :, :], vmin=-clip, vmax=clip, aspect="auto", cmap="gray", interpolation="none")

    plt.subplot(1, 2, 2)
    plt.xlim(0, wave_field.shape[2] - 1)
    plt.ylim(0, wave_field.shape[0] - 1)
    im2 = plt.imshow(wave_field[:, 0, :], vmin=-clip, vmax=clip, aspect="auto", cmap="gray")

    def updatefig(i, *args, **kwargs):
        im1.set_array(wave_field[i, :, :])
        im2.set_array(np.append(wave_field[:i, 0, :], np.zeros((wave_field.shape[0]-i, wave_field.shape[2])), axis=0))
        return im1, im2

    ani = manimation.FuncAnimation(fig, updatefig, frames=wave_field.shape[0], interval=10000.0/60.0, blit=True)
    # writer = manimation.ImageMagickFileWriter()
    # ani.save('wave.gif', writer=writer)

    plt.show()

if __name__ == "__maine__":
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.animation as manimation
    import time

    dt = 1.0/2400.0
    dx = 6.25

    v = np.ones(601, dtype=float)*3000.0

    nt = 8000
    skip = 40

    ts = np.arange(-128, 129)*dt
    freq = 20.0
    s = np.zeros(nt, dtype=float)
    s[:len(ts)] = (1.0 - 2.0*(np.pi*freq*ts)**2.0)*np.exp(-(np.pi*freq*ts)**2.0)
    # s[:len(ts)] = -0.5*np.exp(-(np.pi*freq*ts)**2.0)/(np.pi*freq)**2

    sources = [s]
    # sx = [int((v.shape[0]-1)//2)]
    sx = [0]

    t0 = time.clock()
    wave_field = simulate_1d_wave_equation(v, dx, dt, nt, sources, sx, skip)
    t1 = time.clock()
    print(t1-t0)

    fig = plt.figure()
    plt.ylim(-np.max(np.abs(wave_field)), np.max(np.abs(wave_field)))
    line, = plt.plot(wave_field[0])

    def updatefig_1d(i, *args, **kwargs):
        line.set_ydata(wave_field[i])
        return (line,)

    ani = manimation.FuncAnimation(fig, updatefig_1d, frames=wave_field.shape[0], interval=1000.0/60.0*5.0, blit=True)
    # writer = manimation.ImageMagickFileWriter()
    # ani.save('wave.gif', writer=writer)

    plt.show()
