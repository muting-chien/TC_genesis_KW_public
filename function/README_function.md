In this directory, two files with functions used for analysis are included. The two files are:

1. mjo_mean_state_diagnostic.py
Include more general functions (e.g., calculating the meridional mean, separating 4 seasons...)

2. KW_diagnostic.py
KW-tailored diagnostic functions used in this paper such as:
- generating wavenumber-frequency diagram
- KW-wavenumber-frequency space-time filter
- KW meridional projection
- KW phase composite of 2D and 3D variables
- calculating and plotting EOF of diabatic heating
- rotating EOF
- find the vertical mode for zonal wind from EOF of diabatic heating
- calculating coherence squared between two variables
- calculating theoretical KW phase speed for the first and second mode (Cp1, Cp2) 
#
A third file "create_my_colormap.py" is to create a red-white-blue- colormap. This is used for plotting.
