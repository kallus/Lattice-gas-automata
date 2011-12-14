import scipy.io
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

def draw_state(ax, point, state, write_state=False):
	x = point[0]
	y = point[1]
	ax0.plot(x,y,'ok',markersize=3)
	ax0.plot((x-0.5,x+0.5,x+0.5,x-0.5, x-0.5),(y-0.5,y-0.5,y+0.5,y+0.5,y-0.5),'k')

	if write_state:
		ax.annotate(r'$n_{' + '{0}{1}'.format(2-y,x+1) + r'}' + '$ = ({0},{1},{2},{3})'.format(state[0],state[1],state[2],state[3]), (x-0.45,y+0.35), size='x-small')

	for i in state:
		if 1==state[0]:
			ax0.arrow(x,y+0.1,0,+0.15,head_width=0.05,color=(0,0,0),lw=0.5)
		if 1==state[1]:
			ax0.arrow(x+0.1,y,+0.15,0,head_width=0.05,color=(0,0,0),lw=0.5)
		if 1==state[2]:
			ax0.arrow(x,y-0.1,0,-0.15,head_width=0.05,color=(0,0,0),lw=0.5)
		if 1==state[3]:
			ax0.arrow(x-0.1,y,-0.15,0,head_width=0.05,color=(0,0,0),lw=0.5)


# square-state.pdf figure
fig0 = plt.figure(1, figsize=(7/2.5, 7/2.5), frameon=False)
fig0.subplots_adjust(left=0.0, right=1, top=1, bottom=0)


ax0 = fig0.add_subplot(1,1,1, frameon=False)

draw_state(ax0, (0,0), (1,1,1,1), write_state=True)
draw_state(ax0, (1,0), (1,0,0,1), write_state=True)
draw_state(ax0, (1,1), (0,0,0,1), write_state=True)
draw_state(ax0, (0,1), (0,0,0,0), write_state=True)

ax0.axis((-0.7, 1.7, -0.7, 1.7))
ax0.set_xticks([])
ax0.set_yticks([])

#plt.show()

pp = matplotlib.backends.backend_pdf.PdfPages('figs/square-state.pdf')
pp.savefig(fig0)
pp.close()



# propagation.pdf figure
fig0 = plt.figure(1, figsize=(7/2.5, 7/2.5), frameon=False)

ax0 = fig0.add_subplot(1,1,1, frameon=False)

draw_state(ax0, (0,0), (1,1,1,1), write_state=True)
draw_state(ax0, (1,0), (1,0,0,1), write_state=True)
draw_state(ax0, (1,1), (0,0,0,1), write_state=True)
draw_state(ax0, (0,1), (0,0,0,0), write_state=True)

ax0.axis((-0.7, 1.7, -0.7, 1.7))
ax0.set_xticks([])
ax0.set_yticks([])

#plt.show()

pp = matplotlib.backends.backend_pdf.PdfPages('figs/square-state.pdf')
pp.savefig(fig0)
pp.close()