import os

for fname in os.listdir("."):
    new_name = '0' + fname[1:]
    os.rename(fname, new_name)
    print("{}".format(new_name))


