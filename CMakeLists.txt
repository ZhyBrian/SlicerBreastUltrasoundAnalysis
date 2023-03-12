cmake_minimum_required(VERSION 3.16.3...3.19.7 FATAL_ERROR)

project(BUS_Diagnosis)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "https://github.com/ZhyBrian/SlicerAIBreastUSDiagnosis")
set(EXTENSION_CATEGORY "Utilities")
set(EXTENSION_CONTRIBUTORS "Yi Zhang (Chongqing University, China), Xiaojun Chen (Shanghai Jiao Tong University, China)")
set(EXTENSION_DESCRIPTION "This module is designed to help physicians diagnose intramammary lesions based on breast ultrasound images with nodules.")
set(EXTENSION_ICONURL "https://raw.githubusercontent.com/ZhyBrian/SlicerAIBreastUSDiagnosis/master/Screenshots/BUS_Diagnosis_icon.png")
set(EXTENSION_SCREENSHOTURLS "https://raw.githubusercontent.com/ZhyBrian/SlicerAIBreastUSDiagnosis/master/Screenshots/overview.png")
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
