@echo off
setlocal enabledelayedexpansion

:: Set common variables
set um_per_pix=0.65
set probe_size=3500
set dr=25
set animal=rEO_06
set analysis=cell_counting
set forceReload=0
set use_multiprocessing=1
set num_chs_per_cell=1
set num_images_per_slice=2
set debug_mode=0

:: Define run mode
if !debug_mode! == 1 (
        set debug_flag=--debug
    ) else (
        set debug_flag=
    )

:: Define arrays for hemi and staining values
set hemi_array=left right
if not !num_chs_per_cell! == 1 (
    set staining_suppl=DAPI
    set staining_array=DAPI IBA GFAP NISSL
) else (
    set staining_suppl=""
    set staining_array=DAPI IBA GFAP NISSL 
)

:: Iterate over each hemi and staining value
for %%h in (%hemi_array%) do (
    for %%s in (%staining_array%) do (
        set hemi=%%h
        set staining=%%s
        set output_dir=H:/histology/analyzed_slices/%animal%/multiprocessing/roi_hpc

        :: Set variables based on hemi - crop_coords = top-left, bottom-left, bottom-right and top-right
        if "%%h" == "right" (
            set example_slice_ID=s6_n3
            set slices=s6_n2 s6_n3 s6_n4 s7_n1
            set probe_line_begin=9075 6960
            set probe_line_end=8939 1575
            set crop_coords=1218 3898 1032 5424 2558 5611 2745 4084
        ) else if "%%h" == "left" (
            set example_slice_ID=s7_n2
            set slices=s6_n4 s7_n1 s7_n2 s7_n3
            set probe_line_begin=1878 8139
            set probe_line_end=2096 2754
            set crop_coords=1612 3787 1666 5234 3150 5142 3095 3696
        )

        if !num_images_per_slice! == 1 (
            set example_slice=H:/histology/example_slices/%animal%/%animal%_!example_slice_ID!_color_%%s.png
            set input_dir=H:/histology/new_scans/%animal%/1x
        ) else (
            set example_slice=H:/histology/example_slices/%animal%/%animal%_!example_slice_ID!_color_%%h_%%s.png
            set input_dir=H:/histology/new_scans/%animal%/1x/%%h
        )

        :: Run the Python script with arguments
        echo Running with hemi=%%h, staining=%%s, probe_line_begin=!probe_line_begin!, probe_line_end=!probe_line_end!, crop_coords=!crop_coords!, forceReload=!forceReload!
        python histology_analysis.py ^
         --um_per_pix !um_per_pix! ^
         --probe_size !probe_size! ^
         --hemi %%h --staining %%s ^
         --dr !dr! ^
         --animal !animal! ^
         --analysis !analysis! ^
         --forceReload !forceReload! ^
         --use_multiprocessing !use_multiprocessing! ^
         --input_dir !input_dir! ^
         --output_dir !output_dir! ^
         --example_slice !example_slice! ^
         --slices !slices! ^
         --probe_line_begin !probe_line_begin! ^
         --probe_line_end !probe_line_end! ^
         --crop_coords !crop_coords! ^
         --num_images_per_slice !num_images_per_slice! ^
         --num_chs_per_cell !num_chs_per_cell! ^
         --staining_suppl !staining_suppl! ^
         !debug_flag!
    )
)

endlocal
