@echo off
setlocal enabledelayedexpansion

:: Set common variables
set um_per_pix=0.65
set probe_size=3500
set dr=25
set animal=rEO_05
set analysis=cell_counting
set forceReload=1
set use_multiprocessing=0
set num_chs_per_cell=1
set num_images_per_slice=1
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
    set staining_array=IBA NISSL 
)

:: Iterate over each hemi and staining value
for %%h in (%hemi_array%) do (
    for %%s in (%staining_array%) do (
        set hemi=%%h
        set staining=%%s
        set output_dir=H:/histology/analyzed_slices/%animal%/roi_pyr

        :: Initialize the crop coordinates
        set crop_coords_on=
        set crop_coords_off=

        :: Set variables based on hemi - crop_coords = top-left, bottom-left, bottom-right and top-right
        if "%%h" == "right" (
            set example_slice_ID=s5_n1
            set slices=s5_n2 s5_n1 s6_n2 
            set probe_line_begin=9075 6960
            set probe_line_end=8939 1575

            for %%l in (!slices!) do (
                if "%%l" == "s5_n2" (
                    set crop_coords_on=!crop_coords_on! 3216 2986 8120 3351 8866 3534 8962 3170
                    set crop_coords_off=!crop_coords_off! 6905 3580 7133 3881 7738 3408 7510 3107
                ) else if "%%l" == "s5_n1" (
                    set crop_coords_on=!crop_coords_on! 8216 2920 8120 3285 8866 3469 8962 3104
                    set crop_coords_off=!crop_coords_off! 6905 3580 7133 3881 7738 3408 7510 3107
                ) else if "%%l" == "s6_n2" (
                    set crop_coords_on=!crop_coords_on! 8216 2832 8120 3197 8866 3381 8962 3016
                    set crop_coords_off=!crop_coords_off! 6799 3504 7027 3805 7632 3332 7404 3031
                )
            )
            
        ) else if "%%h" == "left" (
            set example_slice_ID=s5_n1
            set slices=s5_n2 s5_n1 s6_n2 
            set probe_line_begin=1878 8139
            set probe_line_end=2096 2754

            for %%l in (!slices!) do (
                if "%%l" == "s5_n2" (
                    set crop_coords_on=!crop_coords_on! 1624 3616 1849 3919 2459 3453 2234 3150 
                    set crop_coords_off=!crop_coords_off! 2890 3062 2801 3429 3550 3598 3639 3231
                ) else if "%%l" == "s5_n1" (
                    set crop_coords_on=!crop_coords_on! 1401 3726 1658 4002 2212 3470 1955 3194
                    set crop_coords_off=!crop_coords_off! 2890 2992 2801 3359 3550 3528 3639 3161
                ) else if "%%l" == "s6_n2" (
                    set crop_coords_on=!crop_coords_on! 1195 3577 1391 3890 2041 3492 1846 3169
                    set crop_coords_off=!crop_coords_off! 2716 2754 2627 3120 3376 3290 3465 2923
                )
            )
        )

        if !num_images_per_slice! == 1 (
            set example_slice=H:/histology/example_slices/%animal%/%animal%_!example_slice_ID!_color_%%s.png
            set input_dir=H:/histology/new_scans/%animal%/1x
        ) else (
            set example_slice=H:/histology/example_slices/%animal%/%animal%_!example_slice_ID!_color_%%h_%%s.png
            set input_dir=H:/histology/new_scans/%animal%/1x/%%h
        )

        :: Run the Python script with arguments
        echo Running with hemi=%%h, staining=%%s, forceReload=!forceReload!

        python histology_analysis_pyr.py ^
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
        --crop_coords_on "!crop_coords_on!" ^
        --crop_coords_off "!crop_coords_off!" ^
        --num_images_per_slice !num_images_per_slice! ^
        --num_chs_per_cell !num_chs_per_cell! ^
        --staining_suppl !staining_suppl! ^
        !debug_flag!                
    )
)

endlocal
