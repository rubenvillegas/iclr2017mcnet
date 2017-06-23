import os

f = open('train_partition.txt', 'r')
lines = f.readlines()
MAX_NUM_VIDS = 70000

for i in xrange(MAX_NUM_VIDS):
  name = 'sports-1m_{0:09d}'.format(i)
  os.system('pytube -e mp4 -r 360p -f ' + name + ' ' + lines[i].split(' ')[0])

print 'done.'
