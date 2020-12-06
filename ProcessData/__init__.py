import h5py as h5

f=h5.File('../datasets/JN_2017-2020_Meteorology_By_Day.h5','r')

print(f['date'][...])