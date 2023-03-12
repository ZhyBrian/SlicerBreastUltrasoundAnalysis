cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BUS_Diagnosis)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://www.slicer.org/wiki/Documentation/Nightly/Extensions/BUS_Diagnosis")
set(EXTENSION_CATEGORY "Utilities")
set(EXTENSION_CONTRIBUTORS "Yi Zhang, Xiaojun Chen (Shanghai Jiao Tong University)")
set(EXTENSION_DESCRIPTION "This module is designed to help physicians diagnose intramammary lesions based on breast ultrasound images with nodules.")
set(EXTENSION_ICONURL "https://www.example.com/Slicer/Extensions/BUS_Diagnosis.png")
set(EXTENSION_SCREENSHOTURLS "https://www.example.com/Slicer/Extensions/BUS_Diagnosis/Screenshots/1.png")
set(EXTENSION_DEPENDS "NA") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(BUS_Diagnosis)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
