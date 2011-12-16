import scipy.io
import numpy as np
import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

def draw_state(ax, point, state, write_state=False, width=0, height=-0):
	x = point[0]
	y = point[1]
	ax.plot(x,y,'ok',markersize=3)
	ax.plot((x-0.5,x+0.5,x+0.5,x-0.5, x-0.5),(y-0.5,y-0.5,y+0.5,y+0.5,y-0.5),'k')

	if write_state:
		assert(width > 0)
		assert(height > 0)
		ax.annotate(r'$n_{' + '{0}{1}'.format(height-y,x+width-1) + r'}' + '$ = ({0},{1},{2},{3})'.format(state[0],state[1],state[2],state[3]), (x-0.45,y+0.35), size='x-small')

	for i in state:
		if 1==state[0]:
			ax.arrow(x,y+0.1,0,+0.15,head_width=0.05,color=(0,0,0),lw=0.5)
		if 1==state[1]:
			ax.arrow(x+0.1,y,+0.15,0,head_width=0.05,color=(0,0,0),lw=0.5)
		if 1==state[2]:
			ax.arrow(x,y-0.1,0,-0.15,head_width=0.05,color=(0,0,0),lw=0.5)
		if 1==state[3]:
			ax.arrow(x-0.1,y,-0.15,0,head_width=0.05,color=(0,0,0),lw=0.5)


# square-state.pdf figure
fig0 = plt.figure(1, figsize=(6/2.5, 6/2.5), frameon=False)
fig0.subplots_adjust(left=0.0, right=1, top=1, bottom=0)


ax0 = fig0.add_subplot(1,1,1, frameon=False)

draw_state(ax0, (0,0), (1,1,1,1), write_state=True, width=2, height=2)
draw_state(ax0, (1,0), (1,0,0,1), write_state=True, width=2, height=2)
draw_state(ax0, (1,1), (0,0,0,1), write_state=True, width=2, height=2)
draw_state(ax0, (0,1), (0,0,0,0), write_state=True, width=2, height=2)

ax0.axis((-0.7, 1.7, -0.7, 1.7))
ax0.set_xticks([])
ax0.set_yticks([])

#plt.show()

pp = matplotlib.backends.backend_pdf.PdfPages('figs/square-state.pdf')
pp.savefig(fig0)
pp.close()


# propagation-before.pdf figure
fig1 = plt.figure(2, figsize=(6/2.5, 6/2.5), frameon=False)
fig1.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0)

ax1 = fig1.add_subplot(1,1,1, frameon=False)
ax1.set_title('Before propagation')

draw_state(ax1, (0,2), (0,0,1,0))
draw_state(ax1, (1,2), (0,0,0,0))
draw_state(ax1, (2,2), (0,0,0,0))

draw_state(ax1, (0,1), (0,0,0,0))
draw_state(ax1, (1,1), (1,1,1,1))
draw_state(ax1, (2,1), (0,0,0,0))

draw_state(ax1, (0,0), (0,1,0,0))
draw_state(ax1, (1,0), (1,1,0,0))
draw_state(ax1, (2,0), (1,0,0,0))



ax1.axis((-0.7, 2.7, -0.7, 2.7))
ax1.set_xticks([])
ax1.set_yticks([])

#plt.show()

pp = matplotlib.backends.backend_pdf.PdfPages('figs/propagation-before.pdf')
pp.savefig(fig1)
pp.close()


# propagation-after.pdf figure
fig2 = plt.figure(3, figsize=(6/2.5, 6/2.5), frameon=False)
fig2.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0)

ax2 = fig2.add_subplot(1,1,1, frameon=False)
ax2.set_title('After propagation')


draw_state(ax2, (0,2), (0,0,0,0))
draw_state(ax2, (1,2), (1,0,0,0))
draw_state(ax2, (2,2), (0,0,0,0))

draw_state(ax2, (0,1), (0,0,1,1))
draw_state(ax2, (1,1), (1,0,0,0))
draw_state(ax2, (2,1), (1,1,0,0))

draw_state(ax2, (0,0), (0,0,0,0))
draw_state(ax2, (1,0), (0,1,1,0))
draw_state(ax2, (2,0), (0,1,0,0))

ax2.axis((-0.7, 2.7, -0.7, 2.7))
ax2.set_xticks([])
ax2.set_yticks([])

#plt.show()

pp = matplotlib.backends.backend_pdf.PdfPages('figs/propagation-after.pdf')
pp.savefig(fig2)
pp.close()




# collision-before.pdf figure
fig3 = plt.figure(4, figsize=(6/2.5, 6/2.5), frameon=False)
fig3.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0)

ax3 = fig3.add_subplot(1,1,1, frameon=False)
ax3.set_title('Before collision')

draw_state(ax3, (0,2), (1,1,1,0))
draw_state(ax3, (1,2), (0,1,1,0))
draw_state(ax3, (2,2), (1,0,0,0))

draw_state(ax3, (0,1), (0,0,0,0))
draw_state(ax3, (1,1), (1,0,1,0))
draw_state(ax3, (2,1), (0,0,0,0))

draw_state(ax3, (0,0), (1,1,0,0))
draw_state(ax3, (1,0), (1,1,1,0))
draw_state(ax3, (2,0), (0,1,0,1))



ax3.axis((-0.7, 2.7, -0.7, 2.7))
ax3.set_xticks([])
ax3.set_yticks([])

#plt.show()

pp = matplotlib.backends.backend_pdf.PdfPages('figs/collision-before.pdf')
pp.savefig(fig3)
pp.close()


# collision-after.pdf figure
fig4 = plt.figure(5, figsize=(6/2.5, 6/2.5), frameon=False)
fig4.subplots_adjust(left=0.0, right=1, top=0.9, bottom=0)

ax4 = fig4.add_subplot(1,1,1, frameon=False)
ax4.set_title('After collision')



draw_state(ax4, (0,2), (1,1,1,0))
draw_state(ax4, (1,2), (0,1,1,0))
draw_state(ax4, (2,2), (1,0,0,0))

draw_state(ax4, (0,1), (0,0,0,0))
draw_state(ax4, (1,1), (0,1,0,1))
draw_state(ax4, (2,1), (0,0,0,0))

draw_state(ax4, (0,0), (1,1,0,0))
draw_state(ax4, (1,0), (1,1,1,0))
draw_state(ax4, (2,0), (1,0,1,0))

ax4.axis((-0.7, 2.7, -0.7, 2.7))
ax4.set_xticks([])
ax4.set_yticks([])

#plt.show()

pp = matplotlib.backends.backend_pdf.PdfPages('figs/collision-after.pdf')
pp.savefig(fig4)
pp.close()

