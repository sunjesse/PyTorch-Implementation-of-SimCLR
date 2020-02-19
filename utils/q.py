import os

PATH = "/cluster/projects/bwanggroup/EMBRACE"
OUTPATH = "/cluster/home/jessesun/data_directories.txt"

print("running...")

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

def containsDCM(files):
	for f in files:
		if f.endswith(".dcm"):
			return True
	return False

with open(OUTPATH, 'a') as cout:
	for root, dirs, files in os.walk('/cluster/projects/bwanggroup/EMBRACE'):
		i = find_str(root, "series")
		if i >= 0 and containsDCM(files):
			cout.write(os.path.abspath(root+"\n"))
print("done!")

