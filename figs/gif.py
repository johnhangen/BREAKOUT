import imageio

images = []
for cnt in range(1, 60):
    images.append(imageio.imread(f'figs/breakout_screenshot_{cnt}.png'))
imageio.mimsave('figs/breakout.gif', images)