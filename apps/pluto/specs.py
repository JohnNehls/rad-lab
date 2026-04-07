import rsp.pulse_doppler_radar as pdr

# post-hack to ad9364 specs
DAC_bit = 12
f_min = 70e6
f_max = 6e9
fs_max = 56e6
B_min = 200e3
B_max = 56e6

# calculated specs
range_res = pdr.range_resolution(B_max)
print(f"{range_res=} w/o PC")
"We can get a much wider bandwidth with barker"
