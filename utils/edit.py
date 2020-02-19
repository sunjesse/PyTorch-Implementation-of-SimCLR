import os

INPATH = '/cluster/home/jessesun/emb_ic/data/data.txt'
OUTPATH = '/cluster/home/jessesun/emb_ic/data/2dd.txt'
DPATH = '/cluster/home/jessesun/emb_ic/data/diagnosis.txt'

def find_str(s, char):
    index = 0

    if char in s:
        c = char[0]
        for ch in s:
            if ch == c:
                if s[index:index+len(char)] == char:
                    return index

            index += 1

    return -1

d = []
with open(DPATH, 'r') as f:
    for idx, line in enumerate(f):
        l = line.split(' ')
        d.append(str(l[0]))

with open(OUTPATH, 'a') as cout:
	with open(INPATH, 'r') as f:
		for idx, line in enumerate(f):
			idx = find_str(line, "emb")
			edx = idx
			pid = line[idx:-1]
			c = 0
			for char in pid:
				edx += 1
				if char == '_':
					c += 1
				if c == 2:
					break
			pid = line[idx:edx-1]
			if pid in d:
				cout.write(line)

