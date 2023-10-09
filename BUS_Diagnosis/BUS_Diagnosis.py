import logging
import os
import requests
import time

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import numpy as np
import math
import qt
import copy
import SimpleITK as sitk

try:
  from PIL import Image
  import torch
  import torch.nn as nn
  import torch.utils.model_zoo as model_zoo
  from torchvision import transforms
  import torch.nn.functional as F
except:
  try:
    import PyTorchUtils
  except ModuleNotFoundError as e:
    pass
  else:
    torchLogic = PyTorchUtils.PyTorchUtilsLogic()
    try:
      torch = torchLogic.installTorch(askConfirmation=True)
    except:
      pass   
  slicer.util.pip_install('torch torchvision torchaudio')
  from PIL import Image
  import torch
  import torch.nn as nn
  import torch.utils.model_zoo as model_zoo
  from torchvision import transforms
  import torch.nn.functional as F
finally:
  logging.info(f'From BUS_Diagnosis: torch version: {torch.__version__}')

#
# BUS_Diagnosis
#

class BUS_Diagnosis(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "Breast Ultrasound Analysis"  # TODO: make this more human readable by adding spaces
    self.parent.categories = ["Utilities"]  # TODO: set categories (folders where the module shows up in the module selector)
    self.parent.dependencies = []  # TODO: add here list of module names that this module requires
    self.parent.contributors = ["Xiaojun Chen, Yi Zhang (Shanghai Jiao Tong University)"]  # TODO: replace with "Firstname Lastname (Organization)"
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This module is designed to help physicians diagnose intramammary lesions based on breast ultrasound images with nodules. 
Based on the DICOM ultrasound image imported into 3D Slicer, 
this module can segment the nodule from the image and predict whether the nodule is malignant through its built-in AI algorithm.
See more information in https://github.com/ZhyBrian/SlicerBreastUltrasoundAnalysis
"""
    # TODO: replace with organization, grant and thanks
    self.parent.acknowledgementText = """

"""

    # Additional initialization step after application startup is complete
    # slicer.app.connect("startupCompleted()", registerSampleData)


# 
# Register sample data sets in Sample Data module
# 

# def registerSampleData():
#   """
#   Add data sets to Sample Data module.
#   """
#   # It is always recommended to provide sample data for users to make it easy to try the module,
#   # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

#   import SampleData
#   iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

#   # To ensure that the source code repository remains small (can be downloaded and installed quickly)
#   # it is recommended to store data sets that are larger than a few MB in a Github release.

#   # BenignSample6
#   SampleData.SampleDataLogic.registerCustomSampleDataSource(
#     # Category and sample name displayed in Sample Data module
#     category='BreastUltrasoundAnalysisSampleData',
#     sampleName='BenignSample6',
#     # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
#     # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
#     thumbnailFileName=os.path.join(iconsPath, 'BenignSample6.png'),
#     # Download URL and target file name
#     uris="https://github.com/ZhyBrian/SlicerBreastUltrasoundAnalysis/releases/download/v0.0.1/BenignSample6.nrrd",
#     fileNames='BenignSample6.nrrd',
#     # Checksum to ensure file integrity. Can be computed by this command:
#     #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
#     checksums = 'b17ce71e6cbaa6e84edfae9e3ebdabe0155f402ccfd8b074e72f301cb34e50e5',
#     # This node name will be used when the data set is loaded
#     nodeNames='BenignSample6'
#   )

# import hashlib
# print(hashlib.sha256(open(r"C:\Users\84497\Desktop\BenignSample6.nrrd", "rb").read()).hexdigest())

#
# BUS_DiagnosisWidget
#

class BUS_DiagnosisWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False
    self.processingDiag = None
    self.progressDiag = None
    self.saveCount = None

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/BUS_Diagnosis.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = BUS_DiagnosisLogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # developer area
    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUIinput)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUIoutput)
    self.ui.outputMaskSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUIoutputMask)
    self.ui.segmentAllCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUIcheck)

    self.processingDiag = qt.QDialog(slicer.util.mainWindow())
    self.processingDiag.setWindowModality(2)
    self.processingDiag.setWindowTitle("Processing...")
    self.processingDiag.setFixedSize(250, 20)
    self.processingDiag.close()
    
    self.progressDiag = qt.QProgressDialog(slicer.util.mainWindow())
    self.progressDiag.setWindowTitle("Processing...")
    self.progressDiag.setLabelText("Please Wait...")
    self.progressDiag.setWindowModality(2)
    self.progressDiag.setMinimum(0)
    self.progressDiag.setMaximum(1)
    self.progressDiag.setFixedSize(380, 50)
    self.progressDiag.setAutoClose(False)
    self.progressDiag.setCancelButton(None)
    self.progressDiag.close()
    
    self.saveCount = 0
    
    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.ui.installPytorchButton.connect('clicked(bool)', self.onInstallPytorchButton)
    self.ui.pushButtonDownloadSample.connect('clicked(bool)', self.onPushButtonDownloadSample)
    self.ui.pushButtonHideSeg.connect('clicked(bool)', self.onPushButtonHideSeg)
    self.ui.pushButtonShowSeg.connect('clicked(bool)', self.onPushButtonShowSeg)
    self.ui.movetoOffsetButton.connect('clicked(bool)', self.onMovetoOffsetButton)
    self.ui.pushButtonSaveResults.connect('clicked(bool)', self.onPushButtonSaveResults)
    self.ui.pushButtonSetLB.connect('clicked(bool)', self.onPushButtonSetLB)
    self.ui.pushButtonSetRB.connect('clicked(bool)', self.onPushButtonSetRB)
    self.ui.pushButtonResetLB.connect('clicked(bool)', self.ResetLeftOffsetBound)
    self.ui.pushButtonResetRB.connect('clicked(bool)', self.ResetRightOffsetBound)

    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()


  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    # developer area
    # Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None and self.hasObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode):
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # developer area
    # Update node selectors 
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.outputMaskSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputMask"))
    self.ui.segmentAllCheckBox.checked = (self._parameterNode.GetParameter("SegmentAll") == "true")

    # Update buttons states and tooltips
    self.ui.installPytorchButton.toolTip = "Install up-to-date Pytorch(cpu version) to enable this module"
    self.ui.pushButtonDownloadSample.toolTip = "Download benign sample data from GitHub"
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume") and self._parameterNode.GetNodeReference("OutputMask"):
      if self.ui.segmentAllCheckBox.checked:
        self.ui.applyButton.toolTip = "Predict segmentation results for multiple slices"
      else:
        self.ui.applyButton.toolTip = "Predict segmentation and diagnosis results only for the current slice"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and output nodes first"
      self.ui.applyButton.enabled = False
    
    if self.ui.segmentAllCheckBox.checked:
      self.ui.offsetBoundCollapsibleButton.enabled = True
      self.ui.offsetBoundCollapsibleButton.checked = True
      self.ui.pushButtonSetLB.enabled = True
      self.ui.pushButtonSetRB.enabled = True
    else:
      self.ui.offsetBoundCollapsibleButton.checked = False
      self.ui.offsetBoundCollapsibleButton.enabled = False
      
    self.UpdateShowHideButtonStatus()
    self.UpdateMovetoOffsetButtonStatus()
    self.UpdatePushButtonSaveResultsStatus()

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False
    
    
  def updateParameterNodeFromGUIinput(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    # developer area
    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)
    
    slicer.util.setSliceViewerLayers(background=self.ui.inputSelector.currentNodeID)
    slicer.util.resetSliceViews()
    self.onPushButtonHideSeg()

    self.ResetLeftOffsetBound()
    self.ResetRightOffsetBound()
    

  def updateParameterNodeFromGUIoutput(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    # developer area
    self._parameterNode.SetNodeReferenceID("OutputVolume", self.ui.outputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)
    
  def updateParameterNodeFromGUIoutputMask(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    # developer area
    self._parameterNode.SetNodeReferenceID("OutputMask", self.ui.outputMaskSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)
    
    self.onPushButtonHideSeg()
  
  def updateParameterNodeFromGUIcheck(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """

    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    # developer area
    self._parameterNode.SetParameter("SegmentAll", "true" if self.ui.segmentAllCheckBox.checked else "false")
    if self.ui.segmentAllCheckBox.checked:
      qt.QMessageBox.information(slicer.util.mainWindow(), "Caution!", "Caution! It may take several minutes to compute.\
                                 \nAnd the classification result won't be provided in this mode.\
                                   \nFunction of save diagnosis results is also unavailable in this mode.")

    self._parameterNode.EndModify(wasModified)
      

  def onApplyButton(self):
    """
    Run processing when user clicks "Apply" button.
    """
    with slicer.util.tryWithErrorDisplay("Failed to compute results.", waitCursor=True):
      
      isload = self.logic.setupNet(self.progressDiag)
      if not isload:
        qt.QMessageBox.information(slicer.util.mainWindow(), "Warning", "Net weight failed to be loaded. \nAI is unable to provide appropriate diagnosis results.")
        return
      
      self.progressDiag.setWindowTitle("Processing...")
      self.progressDiag.setLabelText("Please Wait...")
      self.progressDiag.show()
      # Compute output
      isSegmentMoreThanOneSlice = self.logic.process(self.ui.inputSelector.currentNode(), 
          self.ui.outputSelector.currentNode(), self.ui.outputMaskSelector.currentNode(), 
          self.progressDiag, self.ui.segmentAllCheckBox.checked, self.ui.labelLB.text, self.ui.labelRB.text)
      
      if isSegmentMoreThanOneSlice == -1:
        self.progressDiag.close()
        return
      
      self.UpdateClassificationResults(isSegmentMoreThanOneSlice)
      self.UpdateShowHideButtonStatus()
      self.onPushButtonShowSeg()
      
      self.UpdateMovetoOffsetButtonStatus()
      self.UpdatePushButtonSaveResultsStatus()

      self.progressDiag.close()
  

  def onInstallPytorchButton(self):
    slicer.util.pip_install('torch torchvision torchaudio')
    import torch
    logging.info(f'From BUS_Diagnosis: torch version: {torch.__version__}')
  
  def onPushButtonDownloadSample(self):
    directory = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Choose a Folder", "./")
    if os.path.exists(directory):
      url = "https://files.zohopublic.com.cn/public/workdrive-public/download/jy075c6237954580e4ccf98fca3fd55bacf66?x-cli-msg=%7B%22isFileOwner%22%3Afalse%2C%22version%22%3A%221.0%22%7D"
      name = "BenignSample.nrrd"
      if not os.path.exists(os.path.join(directory, name)):
        self.progressDiag.setWindowTitle("Downloading sample data...")
        self.progressDiag.show()
        self.logic.downloadShowProgress(url, directory, name, self.progressDiag, showsuccess=0)
        self.progressDiag.close()
      if os.path.exists(os.path.join(directory, name)):
        slicer.util.loadVolume(os.path.join(directory, name))
      
  
  def onPushButtonHideSeg(self):
    if self.logic.segmentationNode is not None:
      self.logic.segmentationNode.SetDisplayVisibility(False)
      self.ui.pushButtonHideSeg.enabled = False
      self.ui.pushButtonShowSeg.enabled = True
    
  def onPushButtonShowSeg(self):
    if self.logic.segmentationNode is not None:
      self.logic.segmentationNode.SetDisplayVisibility(True)
      self.ui.pushButtonShowSeg.enabled = False
      self.ui.pushButtonHideSeg.enabled = True
  
  def UpdateShowHideButtonStatus(self):
    if self.logic.segmentationNode is None:
      self.ui.pushButtonHideSeg.toolTip = "Click `AI Automatically Segment and Diagnose` button to generate prediction results first"
      self.ui.pushButtonHideSeg.enabled = False
      self.ui.pushButtonShowSeg.toolTip = "Click `AI Automatically Segment and Diagnose` button to generate prediction results first"
      self.ui.pushButtonShowSeg.enabled = False
    else:
      self.ui.pushButtonHideSeg.toolTip = "Hide segmentation mask displayed in the scene"
      self.ui.pushButtonHideSeg.enabled = True
      self.ui.pushButtonShowSeg.toolTip = "Show segmentation mask in the scene"
      self.ui.pushButtonShowSeg.enabled = True
  
  def onMovetoOffsetButton(self):
    try:
      offset = float(self.ui.labelOffset.text)
    except:
      pass
    else:
      bounds = [0,] * 6
      slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceBounds(bounds)
      redLowerBound = bounds[4]
      redUpperBound = bounds[5]
      if offset >= redLowerBound and offset <= redUpperBound:
        slicer.app.layoutManager().sliceWidget("Red").sliceLogic().SetSliceOffset(offset)
      else:
        qt.QMessageBox.information(slicer.util.mainWindow(), "Error!", "Error! Offset out of bounds.     ")
  
  def UpdateMovetoOffsetButtonStatus(self):
    if self.ui.inputSelector.currentNode() is not None:
      self.ui.movetoOffsetButton.enabled = True
    else:
      self.ui.movetoOffsetButton.enabled = False
    
    
  def onPushButtonSaveResults(self):
    directory = qt.QFileDialog.getExistingDirectory(slicer.util.mainWindow(), "Choose a Folder", "./")
    if os.path.exists(directory):
      self.processingDiag.show()
      
      inputVolumeNode = slicer.mrmlScene.GetNodeByID(self.ui.labelInputNode.text)
      outputVolumeNode = slicer.mrmlScene.GetNodeByID(self.ui.labelOutputNode.text)
      segmentNode = slicer.mrmlScene.GetNodeByID(self.ui.labelSegNode.text)
      self.logic.currentSegmentID = segmentId = segmentNode.GetSegmentation().GetNthSegmentID(0)
      segmentArray = slicer.util.arrayFromSegmentBinaryLabelmap(segmentNode, segmentId, inputVolumeNode)
      inputVolumeArray = slicer.util.arrayFromVolume(inputVolumeNode)
      outputVolumeArray = slicer.util.arrayFromVolume(outputVolumeNode)
      
      inputVolumeArray = np.transpose(inputVolumeArray, axes=(1, 2, 0))
      h, w, c = inputVolumeArray.shape
      inputVolumeArray = inputVolumeArray[:, :, self.logic.currentSlice]
      outputVolumeArray = np.transpose(outputVolumeArray, axes=(1, 2, 0))
      outputVolumeArray = outputVolumeArray[:, :, self.logic.currentSlice]
      segmentArray = np.transpose(segmentArray, axes=(1, 2, 0))
      segmentArray = segmentArray[:, :, self.logic.currentSlice]
      
      shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
      seriesItem = shNode.GetItemByDataNode(inputVolumeNode)
      studyItem = shNode.GetItemParent(seriesItem)
      patientItem = shNode.GetItemParent(studyItem)
      patientName = self.ui.labelPatientName.text
      
      savePath = os.path.join(directory, f"Results_{patientName}_1")
      self.saveCount += 1
      if not os.path.exists(savePath):
        os.makedirs(savePath)
      else:
        while os.path.exists(savePath):
          savePath = savePath[:-1] + f"{int(savePath[-1]) + 1}"
        os.makedirs(savePath)
      
      sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(inputVolumeArray), sitk.sitkUInt8), os.path.join(savePath, f"Image.png"))
      sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(np.flip(np.flip(outputVolumeArray, axis=0), axis=1)), sitk.sitkUInt8), os.path.join(savePath, f"Label_AIpredicted.png"))
      sitk.WriteImage(sitk.Cast(sitk.GetImageFromArray(segmentArray * 255), sitk.sitkUInt8), os.path.join(savePath, f"Label_Revised.png"))
      with open(os.path.join(savePath, f"Classification_Result.txt"), 'w') as f:
        f.write(f"Patient Name       :  {patientName}\n")
        f.write(f"Patient ID         :  {shNode.GetItemAttribute(patientItem, 'DICOM.PatientID')}\n")
        f.write(f"Birth Date         :  {shNode.GetItemAttribute(patientItem, 'DICOM.PatientBirthDate')}\n")
        f.write(f"Patient Sex        :  {shNode.GetItemAttribute(patientItem, 'DICOM.PatientSex')}\n\n")
        f.write(f"Study Date         :  {shNode.GetItemAttribute(studyItem, 'DICOM.StudyDate')}\n")
        f.write(f"Study ID           :  {shNode.GetItemAttribute(studyItem, 'DICOM.StudyID')}\n")
        f.write(f"Study Description  :  {shNode.GetItemAttribute(studyItem, 'DICOM.StudyDescription')}\n\n")
        f.write(f"Series Number      :  {shNode.GetItemAttribute(seriesItem, 'DICOM.SeriesNumber')}\n")
        # f.write(f"Series Description :  {shNode.GetItemAttribute(seriesItem, 'DICOM.SeriesDescription')}\n")
        f.write(f"Modality           :  {shNode.GetItemAttribute(seriesItem, 'DICOM.Modality')}\n")
        f.write(f"Size (hxw)         :  {h}x{w}\n")
        f.write(f"Count              :  {c}\n\n")
        f.write(f"Offset             :  {self.ui.labelOffset.text}\n")
        f.write(f"Slice Index        :  {self.logic.currentSlice}\n")
        f.write(f"AI Predicted Class :   {self.ui.labelClass.text}\n")
        f.write(f"Probability of {self.ui.labelProb.text}")
      
      self.processingDiag.close()
      # qt.QMessageBox.information(slicer.util.mainWindow(), "Information", "Diagnosis Results successfully saved!     ")
    elif directory == "":
      pass
    else:
      qt.QMessageBox.information(slicer.util.mainWindow(), "Error", "The selected directory does not exist!     ")
      
  def UpdatePushButtonSaveResultsStatus(self):
    if (self.logic.segmentationNode is not None) and self.ui.labelOffset.text != "":
      self.ui.pushButtonSaveResults.enabled = True
    else:
      self.ui.pushButtonSaveResults.enabled = False
  
  
  def UpdateClassificationResults(self, isSegmentMoreThanOneSlice):
    self.ui.labelInputNode.setText(self.ui.inputSelector.currentNodeID)
    self.ui.labelOutputNode.setText(self.ui.outputSelector.currentNodeID)
    self.ui.labelSegNode.setText(self.logic.segmentationNode.GetID())
    
    shNode = slicer.mrmlScene.GetSubjectHierarchyNode()
    seriesItem = shNode.GetItemByDataNode(slicer.mrmlScene.GetNodeByID(self.ui.labelInputNode.text))
    studyItem = shNode.GetItemParent(seriesItem)
    patientItem = shNode.GetItemParent(studyItem)
    patientName = shNode.GetItemAttribute(patientItem, 'DICOM.PatientName') if \
          shNode.GetItemAttribute(patientItem, 'DICOM.PatientName') != "" else "UnknownPatient"
    self.ui.labelPatientName.setText(patientName)
    
    if isSegmentMoreThanOneSlice:
      self.ui.labelClass.setText("")
      self.ui.labelProb.setText("")
      self.ui.labelOffset.setText("")
    else:
      self.ui.labelClass.setText(self.logic.tumorClass)
      self.ui.labelProb.setText(f"{self.logic.tumorClass}:  {self.logic.tumorProb: .4f}%")
      self.ui.labelOffset.setText(f"{self.logic.currentOffset}")
      if self.logic.tumorClass == "Benign":
        self.ui.labelClass.setStyleSheet("color:green;")
        self.ui.labelProb.setStyleSheet("color:green;")
      elif self.logic.tumorClass == "Malignant":
        self.ui.labelClass.setStyleSheet("color:red;")
        self.ui.labelProb.setStyleSheet("color:red;")
    
  def onPushButtonSetLB(self):
    self.ui.labelLB.setText(f'{slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset():.4f}')
    
  def onPushButtonSetRB(self):
    self.ui.labelRB.setText(f'{slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset():.4f}')
  
  def ResetLeftOffsetBound(self):
    self.ui.labelLB.setText("default")
  
  def ResetRightOffsetBound(self):
    self.ui.labelRB.setText("default")
  


#
# BUS_DiagnosisLogic
#

class BUS_DiagnosisLogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.model_seg = ConvFormer_MTL().to(self.device).eval()
    self.weight_path_seg = os.path.join(os.path.dirname(__file__), 'Resources/net_weight_base.pth')
    
    self.weightloaded = 0

    self.transform = transforms.Compose([transforms.ToTensor()]) 
    self.wh = 256
    self.segmentationNode = None
    self.currentSegmentID = None
    self.currentOffset = 0
    self.currentSlice = 0
    self.redLowerBound = 0
    self.redUpperBound = 0
    
    self.isMultipleChannel = False
    self.numberOfChannels = 1
    self.segmentAll = False
    self.tumorClass = 'Unknown'
    self.tumorProb = -1

  def setDefaultParameters(self, parameterNode):
    # developer area
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("SegmentAll"):
      parameterNode.SetParameter("SegmentAll", "false")
  
  
  def downloadShowProgress(self, url, path, name, progressDiag, showsuccess=1):
    if not os.path.exists(path):   
      os.mkdir(path)
    start = time.time() 
    requests.DEFAULT_RETRIES = 5  
    s = requests.session()
    s.keep_alive = False  
    headers = { 'Connection': 'close',
               'Accept-Encoding': 'identity',}
    response = requests.get(url, stream=True, verify=False, headers=headers)
    size = 0    
    chunk_size = 1024  
    content_size = int(response.headers['content-length'])  
    try:
      if response.status_code == 200:   
        progressDiag.setLabelText(f"File size: {content_size/chunk_size/1024:.2f} MB")
        progressDiag.setMaximum(100)
        filepath = os.path.join(path, name)
        with open(filepath, 'wb') as file:   
          for data in response.iter_content(chunk_size = chunk_size):
            file.write(data)
            size += len(data)
            progressDiag.setValue(float(size / content_size * 100))
      end = time.time()   
      if showsuccess:
        qt.QMessageBox.information(slicer.util.mainWindow(), "Information", f"Download successfully! Time spent: {end - start:.2f}s     ")
    except:
      qt.QMessageBox.information(slicer.util.mainWindow(), "Error", "Download failed!     ")
  
  
  def setupNet(self, progressDiag):
    if not self.weightloaded:
      if not os.path.exists(self.weight_path_seg):
        url = "https://files.zohopublic.com.cn/public/workdrive-public/download/jy075dd560e5a2fe7475b8a02ebc889aca769?x-cli-msg=%7B%22isFileOwner%22%3Afalse%2C%22version%22%3A%221.0%22%7D"
        path = os.path.join(os.path.dirname(__file__), 'Resources')
        name = "net_weight_base.pth"
        
        progressDiag.setWindowTitle("Downloading model (only for once)...")
        progressDiag.show()
        self.downloadShowProgress(url, path, name, progressDiag, showsuccess=0)
        progressDiag.close()
      
      try:
        self.model_seg.load_state_dict(torch.load(self.weight_path_seg, map_location=torch.device(self.device)))
      except:
        logging.info('Net weight failed to be loaded!')
        self.weightloaded = 0
        return 0
      else:
        logging.info('Net successfully loaded weight!')
        self.weightloaded = 1
        return 1
    else:
      return 1
        


  def keep_image_size_open_mask_test(self, input_img, size, scale_method='ANTIALIAS'):
    logging.info(f'input image size: {input_img.shape}')
    img = Image.fromarray(np.float32(input_img))

    temp = max(img.size)
    mask = Image.new('L', (temp, temp))
    mask.paste(img, (0, 0))
    if scale_method == 'ANTIALIAS':
        mask = mask.resize(size, Image.LANCZOS) 
    else:
        mask = mask.resize(size, Image.NEAREST) 
    return mask, img.size[0], img.size[1]
  
  
  def fill_binary_fig_holes_and_eliminate_noise(self, blank_thresh):
    sitk_blank_thresh = sitk.GetImageFromArray(blank_thresh)
    
    median = sitk.MedianImageFilter()
    median.SetRadius(2)
    sitk_blank_thresh = median.Execute(sitk_blank_thresh)
    
    sitk_blank_thresh_copy = copy.deepcopy(sitk_blank_thresh)
    sitk_blank_thresh_copy = sitk.ConnectedThreshold(sitk_blank_thresh_copy, seedList=[(0, 0)], lower=0, upper=250, replaceValue=255)
    sitk_blank_thresh_copy_inv = sitk.InvertIntensity(sitk_blank_thresh_copy)
    sitk_blank_thresh_out = sitk.Or(sitk_blank_thresh, sitk_blank_thresh_copy_inv)
    
    sitk_label_image = sitk.ConnectedComponent(sitk_blank_thresh_out, False)
    
    blank_thresh_out = sitk.GetArrayFromImage(sitk_blank_thresh_out)
    label_image = sitk.GetArrayFromImage(sitk_label_image)
    label_mask = copy.deepcopy(label_image)
    num_connected = np.max(label_image) + 1
    if num_connected <= 2:
      pass
    else:
      areas = [0, ]
      for i in range(1, num_connected):
        areas.append(np.sum(label_image == i))
      maxArea = max(areas)
      for j in range(1, num_connected):
        currentArea = areas[j]
        if currentArea < 0.1 * maxArea:
          label_mask[label_mask == j] = 0
      blank_thresh_out = blank_thresh_out * label_mask
      blank_thresh_out[blank_thresh_out > 255] = 255
          
    return blank_thresh_out
  
  
  def AIAssistedSegment(self, input_img):
    input_img, w, h = self.keep_image_size_open_mask_test(input_img, size=(self.wh, self.wh))
    input_img = torch.cat([self.transform(input_img), self.transform(input_img), self.transform(input_img)], dim = 0)
    input_img = torch.unsqueeze(input_img, dim=0).to(self.device)
    result_img, result_cls = self.model_seg(input_img)
    result_img = transforms.ToPILImage()(result_img[0].float())
    result_img = result_img.crop(box=(0, 0, w/(max(w,h)/self.wh), h/(max(w,h)/self.wh)))  
    result_img = result_img.resize((w, h), Image.LANCZOS)
    result_img = np.array(result_img)
    result_img[result_img >= 255*0.5] = 255
    result_img[result_img < 255*0.5] = 0
    result_img = self.fill_binary_fig_holes_and_eliminate_noise(result_img)
    result_img = np.expand_dims(result_img, axis=2)
    
    return result_img, result_cls
  
  def fromClsResulttoClassProb(self, result_cls):
    if torch.squeeze(result_cls)[0].item() > torch.squeeze(result_cls)[1].item():
      return 'Benign', torch.squeeze(result_cls)[0].item() * 100
    else:
      return 'Malignant', torch.squeeze(result_cls)[1].item() * 100
  
  def fromCurrentOffsettoCurrentSlice(self, currentOffset, LowerBound, UpperBound, numberOfChannels):
    minOffset = LowerBound + ((UpperBound - LowerBound) / (2 * numberOfChannels))
    gapValue = (UpperBound - LowerBound) / numberOfChannels
    
    return round((currentOffset - minOffset) / gapValue)
 
    
  def process(self, inputVolume, outputVolume, outputMask, progressDiag, SegmentAll=False, leftOffsetBound="default", rightOffsetBound="default"):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be segmented and classified
    :param outputVolume: segmentation result
    :param SegmentAll: if True then all the slice will be segmented simultaneously, otherwise only current slice will be segmented
    """  

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    # import time
    startTime = time.time()
    logging.info('Processing started')
    progressDiag.setMaximum(101)
    progressDiag.setValue(0)
    self.segmentAll = SegmentAll

    input_img = slicer.util.arrayFromVolume(inputVolume)
    if len(input_img.shape) >= 3:
      input_img = np.transpose(input_img, axes=(1, 2, 0))
      if input_img.shape[2] == 1:
        self.isMultipleChannel = False
        self.numberOfChannels = 1
        self.currentOffset = 0.0
        self.currentSlice = 0
        input_img = np.squeeze(input_img, axis=2)
      else:
        self.isMultipleChannel = True
        self.numberOfChannels = input_img.shape[2]
        self.currentOffset = slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceOffset()
        bounds = [0,] * 6
        slicer.app.layoutManager().sliceWidget("Red").sliceLogic().GetSliceBounds(bounds)
        self.redLowerBound = bounds[4]
        self.redUpperBound = bounds[5]
        self.currentSlice = self.fromCurrentOffsettoCurrentSlice(self.currentOffset, 
                      self.redLowerBound, self.redUpperBound, self.numberOfChannels)
        if not self.segmentAll:
          input_img = input_img[:, :, self.currentSlice]
        else:
          if leftOffsetBound == "default":
            leftBound = 0
          else:
            leftBound = self.fromCurrentOffsettoCurrentSlice(float(leftOffsetBound), 
                      self.redLowerBound, self.redUpperBound, self.numberOfChannels)
          if rightOffsetBound == "default":
            rightBound = self.numberOfChannels - 1
          else:
            rightBound = self.fromCurrentOffsettoCurrentSlice(float(rightOffsetBound), 
                      self.redLowerBound, self.redUpperBound, self.numberOfChannels)
          if rightBound < leftBound:
            qt.QMessageBox.information(slicer.util.mainWindow(), "Error", "Left offset bound > right offset bound!    ")
            return -1
    else:
      qt.QMessageBox.information(slicer.util.mainWindow(), "Error", "input should be 3D volume!     ")
      logging.info('input should be 3D volume!')
      return -1
    
    h, w = input_img.shape[0], input_img.shape[1]
    
    if self.isMultipleChannel:
      result_img_temp = np.zeros((h, w, self.numberOfChannels))
      if len(input_img.shape) >= 3:
        assert self.segmentAll == True
        for i in range(leftBound, rightBound + 1):
          result_img_temp[:, :, i] = np.squeeze(self.AIAssistedSegment(input_img[:, :, i])[0], axis=2)
          progressDiag.setValue(100 * (i - leftBound) / (rightBound - leftBound + 0.001))
        result_img = np.transpose(result_img_temp, axes=(2, 0, 1)).astype('int32')
      else:
        result_img, result_cls = self.AIAssistedSegment(input_img)  
        self.tumorClass, self.tumorProb = self.fromClsResulttoClassProb(result_cls)
        result_img_temp[:, :, self.currentSlice] = np.squeeze(result_img, axis=2)
        result_img = np.transpose(result_img_temp, axes=(2, 0, 1)).astype('int32')
    else:
      result_img, result_cls = self.AIAssistedSegment(input_img)  
      self.tumorClass, self.tumorProb = self.fromClsResulttoClassProb(result_cls)
      result_img = np.transpose(result_img, axes=(2, 0, 1)).astype('int32')

    # slicer.mrmlScene.RemoveNode(self.segmentationNode)
    # self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    outputMask.CreateDefaultDisplayNodes()
    if outputMask.GetSegmentation().GetNumberOfSegments() >= 1:
      self.currentSegmentID = outputMask.GetSegmentation().GetNthSegmentID(0)
    else:
      self.currentSegmentID = outputMask.GetSegmentation().AddEmptySegment()
    slicer.util.updateSegmentBinaryLabelmapFromArray(result_img, outputMask, self.currentSegmentID, inputVolume)
    self.segmentationNode = outputMask

    slicer.util.updateVolumeFromArray(outputVolume, np.flip(np.flip(result_img, axis=1), axis=2))
    progressDiag.setValue(101)

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
    
    if self.isMultipleChannel and self.segmentAll:
      return 1   # isSegmentMoreThanOneSlice = 1
    else:
      return 0   # isSegmentMoreThanOneSlice = 0











class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, act_layer=nn.GELU):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, 1e-6)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale)
        self.norm2 = nn.LayerNorm(dim, 1e-6)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class ConvBlock(nn.Module):

    def __init__(self, inplanes, outplanes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1):
        super(ConvBlock, self).__init__()

        expansion = 4
        med_planes = outplanes // expansion

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(med_planes, eps=1e-6)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(med_planes, eps=1e-6)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, outplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(outplanes, eps=1e-6)
        self.act3 = act_layer(inplace=True)

        if res_conv:
            self.residual_conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.residual_bn = nn.BatchNorm2d(outplanes, eps=1e-6)

        self.res_conv = res_conv

    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t=None, return_x_2=True):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x) if x_t is None else self.conv2(x + x_t)
        x = self.bn2(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)

        if self.res_conv:
            residual = self.residual_conv(residual)
            residual = self.residual_bn(residual)

        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x


class FCUDown(nn.Module):

    def __init__(self, inplanes, outplanes, dw_stride, act_layer=nn.GELU):
        super(FCUDown, self).__init__()
        self.dw_stride = dw_stride

        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.sample_pooling = nn.AvgPool2d(kernel_size=dw_stride, stride=dw_stride)

        self.ln = nn.LayerNorm(outplanes, 1e-6)
        self.act = act_layer()

    def forward(self, x, x_t):
        x = self.conv_project(x) 

        x = self.sample_pooling(x).flatten(2).transpose(1, 2)
        x = self.ln(x)
        x = self.act(x)

        x = torch.cat([x_t[:, 0][:, None, :], x], dim=1)

        return x


class FCUUp(nn.Module):

    def __init__(self, inplanes, outplanes, up_stride, act_layer=nn.ReLU):
        super(FCUUp, self).__init__()

        self.up_stride = up_stride
        self.conv_project = nn.Conv2d(inplanes, outplanes, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm2d(outplanes, 1e-6)
        self.act = act_layer()

    def forward(self, x, H, W):
        B, _, C = x.shape
        x_r = x[:, 1:].transpose(1, 2).reshape(B, C, H, W)
        x_r = self.act(self.bn(self.conv_project(x_r)))

        return F.interpolate(x_r, size=(H * self.up_stride, W * self.up_stride))


class ConvTransBlock(nn.Module):

    def __init__(self, inplanes, outplanes, res_conv, stride, dw_stride, embed_dim, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, groups=1):

        super(ConvTransBlock, self).__init__()
        expansion = 4
        self.cnn_block = ConvBlock(inplanes=inplanes, outplanes=outplanes, res_conv=res_conv, stride=stride, groups=groups)
        self.fusion_block = ConvBlock(inplanes=outplanes, outplanes=outplanes, groups=groups)
        self.squeeze_block = FCUDown(inplanes=outplanes // expansion, outplanes=embed_dim, dw_stride=dw_stride)
        self.expand_block = FCUUp(inplanes=embed_dim, outplanes=outplanes // expansion, up_stride=dw_stride)
        self.trans_block = Block(
            dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.dw_stride = dw_stride
        self.embed_dim = embed_dim

    def forward(self, x, x_t):
        x, x2 = self.cnn_block(x)

        _, _, H, W = x2.shape

        x_st = self.squeeze_block(x2, x_t)

        x_t = self.trans_block(x_st + x_t)

        x_t_r = self.expand_block(x_t, H // self.dw_stride, W // self.dw_stride)
        x = self.fusion_block(x, x_t_r, return_x_2=False)

        return x, x_t


class Conformer(nn.Module):

    def __init__(self, patch_size=16, in_chans=3, num_classes=1000, base_channel=64, channel_ratio=4,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim 
        assert depth % 3 == 0

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False) 
        self.bn1 = nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        stage_1_channel = int(base_channel * channel_ratio)
        trans_dw_stride = patch_size // 4
        self.conv_1 = ConvBlock(inplanes=64, outplanes=stage_1_channel, res_conv=True, stride=1)
        self.trans_patch_conv = nn.Conv2d(64, embed_dim, kernel_size=trans_dw_stride, stride=trans_dw_stride, padding=0)
        self.trans_1 = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             qk_scale=qk_scale)

        init_stage = 2
        fin_stage = depth // 3 + 1
        for i in range(init_stage, fin_stage):
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        stage_1_channel, stage_1_channel, False, 1, dw_stride=trans_dw_stride, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale
                    )
            )


        stage_2_channel = int(base_channel * channel_ratio * 2)
        init_stage = fin_stage 
        fin_stage = fin_stage + depth // 3
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_1_channel if i == init_stage else stage_2_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_2_channel, res_conv, s, dw_stride=trans_dw_stride // 2, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale
                    )
            )

        stage_3_channel = int(base_channel * channel_ratio * 2 * 2)
        init_stage = fin_stage  
        fin_stage = fin_stage + depth // 3  
        for i in range(init_stage, fin_stage):
            s = 2 if i == init_stage else 1
            in_channel = stage_2_channel if i == init_stage else stage_3_channel
            res_conv = True if i == init_stage else False
            self.add_module('conv_trans_' + str(i),
                    ConvTransBlock(
                        in_channel, stage_3_channel, res_conv, s, dw_stride=trans_dw_stride // 4, embed_dim=embed_dim,
                        num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale
                    )
            )
        self.fin_stage = fin_stage


    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x_out_1 = self.act1(self.bn1(self.conv1(x)))
        x_base = self.maxpool(x_out_1)

        # 1 stage
        x = self.conv_1(x_base, return_x_2=False)

        x_t = self.trans_patch_conv(x_base).flatten(2).transpose(1, 2)
        x_t = torch.cat([cls_tokens, x_t], dim=1)
        x_t = self.trans_1(x_t)
        
        # 2 ~ final 
        for i in range(2, self.fin_stage):
            x, x_t = eval('self.conv_trans_' + str(i))(x, x_t)
            if i == 4:
                x_out_2 = x
            elif i == 8:
                x_out_3 = x
            elif i == 12:
                x_out_4 = x

        return x_out_4, x_out_3, x_out_2, x_out_1, x_t


class BasicConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(BasicConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=False),
        )

class MultiConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, conv_num=2, intermediate_out=False):
        super(MultiConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.intermediate_out = intermediate_out
        assert conv_num >= 2
        
        self.conv1 = BasicConv(in_channels, mid_channels)
        conv_list = []
        for i in range(conv_num - 1):
            if i != conv_num - 2:
                conv_list.append(BasicConv(mid_channels, mid_channels))
            else:
                conv_list.append(BasicConv(mid_channels, out_channels))
        self.conv2 = nn.Sequential(*conv_list)
    
    def forward(self, x):
        x_mid = self.conv1(x)
        x_out = self.conv2(x_mid)
        
        if self.intermediate_out:
            return x_out, x_mid
        else:
            return x_out


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, conv_num=2, skip=True):
        super(Up, self).__init__()
        self.skip = skip
        if self.skip:
            self.mid_channels = out_channels
            self.skip_channels = out_channels
        else:
            self.mid_channels = in_channels
            self.skip_channels = 0
        
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, self.mid_channels, 1, 1),
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, self.mid_channels, kernel_size=2, stride=2)

        self.conv = MultiConv(self.mid_channels + self.skip_channels, out_channels, conv_num=conv_num)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        if x2 is not None:
            assert self.skip == True
            diff_y = x2.size()[2] - x1.size()[2]
            diff_x = x2.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                            diff_y // 2, diff_y - diff_y // 2])

            x1 = torch.cat([x2, x1], dim=1)
        else:
            assert self.skip == False
        
        x = self.conv(x1)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )


class UNet_ConvFormer(nn.Module):
    def __init__(self,
                 in_channels: int = 3,
                 num_classes: int = 1,
                 bilinear: bool = True,
                 base_c: int = 64,
                 size='base',
                 pretrained=False):
        super(UNet_ConvFormer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        if size == 'base':
            self.seg_backbone = Conformer(patch_size=16, channel_ratio=6, embed_dim=576, depth=12,
                      num_heads=9, mlp_ratio=4, qkv_bias=True)

            self.up1 = Up(base_c * 24, base_c * 12, bilinear, conv_num=3)
            self.up2 = Up(base_c * 12, base_c * 6, bilinear, conv_num=3)
            self.up3 = Up(base_c * 6, base_c * 1, bilinear, conv_num=2)
            self.up4 = Up(base_c * 1, base_c, bilinear, conv_num=2, skip=False)
            self.out_conv = OutConv(base_c, num_classes)


    def forward(self, x):
        x_out_4, x_out_3, x_out_2, x_out_1, x_t = self.seg_backbone(x)

        x = self.up1(x_out_4, x_out_3)
        x = self.up2(x, x_out_2)
        x = self.up3(x, x_out_1)
        x = self.up4(x, None)
        logits = self.out_conv(x)

        return logits, x_t


class Classifier(nn.Module):
    def __init__(self, embed_dim, num_classes, 
                 iter_num_dilate=25, iter_num_shrink=15):
        super(Classifier, self).__init__()
        self.trans_norm = nn.LayerNorm(embed_dim)
        self.conv_more = BasicConv(embed_dim, embed_dim)
        self.trans_cls_head = nn.Linear(embed_dim * 4, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((2, 2))
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mtr_attn = MultiRegionAttention(dim=embed_dim, num_heads=8, qkv_bias=True,
                                             iter_num_dilate=iter_num_dilate, 
                                             iter_num_shrink=iter_num_shrink)
    
    def forward(self, x_t, mask=None):
        B, N, C = x_t.shape
        H = W = int(math.sqrt(N - 1))
        x_t = x_t[:, 1:]
        
        x_t = x_t + self.mtr_attn(self.norm1(x_t), mask)
        
        x_t = self.trans_norm(x_t)
        x_r = x_t.transpose(1, 2).reshape(B, C, H, W)
        
        x_r = self.conv_more(x_r)
        out_cls = self.trans_cls_head(torch.flatten(self.avgpool(x_r), 1))
        
        return out_cls


def soft_dilate(img):
    if len(img.shape) == 4:
        return F.max_pool2d(img, (3,3), (1,1), (1,1))
    elif len(img.shape) == 5:
        return F.max_pool3d(img, (3,3,3), (1,1,1), (1,1,1))

def mask_dilate(mask, iter=25):
    dilate = mask
    for i in range(iter):
        dilate = soft_dilate(dilate)
    return dilate

def mask_shrink(mask, iter=5):
    dilate = 1 - mask
    for i in range(iter):
        dilate = soft_dilate(dilate)
    shrink = 1 - dilate
    return shrink

def cnn_mask_to_transformer_att(mask_1, mask_2, scale):
    mask_1 = F.interpolate(mask_1, size=None, scale_factor=scale, mode='bilinear', align_corners=None)
    mask_2 = F.interpolate(mask_2, size=None, scale_factor=scale, mode='bilinear', align_corners=None)
    mask_h = mask_2.flatten(2)
    mask_v = mask_1.flatten(2).transpose(-1, -2)
    mask = (mask_v @ mask_h).unsqueeze(1)
    mask = nn.Hardtanh()(mask + mask.transpose(-1, -2))
    return mask

def combine_att_masks(mask_intra, mask_peri, wbs, scale):
    att_mask_intra = cnn_mask_to_transformer_att(mask_intra, mask_intra, scale) * wbs[0][0] + wbs[0][1]
    att_mask_peri = cnn_mask_to_transformer_att(mask_peri, mask_peri, scale) * wbs[1][0] + wbs[1][1]
    att_mask_list = [att_mask_intra, att_mask_peri, att_mask_intra, att_mask_peri, 
                     att_mask_intra, att_mask_peri, att_mask_intra, att_mask_peri]
    return torch.concat(att_mask_list, dim=1)


class MultiRegionAttention(nn.Module):
    def __init__(self,
                 dim,   
                 num_heads=8,
                 qkv_bias=True,
                 qk_scale=None,
                 iter_num_dilate=25, 
                 iter_num_shrink=15):
        super(MultiRegionAttention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.iter_num_dilate = iter_num_dilate
        self.iter_num_shrink = iter_num_shrink
        self.wb_att = nn.Parameter(torch.Tensor([[1.0, 0.2], [1.0, 0.2]]), requires_grad=True)

    def forward(self, x, mask=None):
        B, N, C = x.shape
        
        if mask is not None:
            mask_intra = mask
            mask_dil = mask_dilate(mask, iter=self.iter_num_dilate)
            mask_shr = mask_shrink(mask, iter=self.iter_num_shrink)

            mask_peri = mask_dil - mask_shr
            scale = int(math.sqrt(N)) / 256
            assert scale == 1 / 16 

            mask_trans_att = combine_att_masks(mask_intra, mask_peri, self.wb_att, scale)
            assert mask_trans_att.shape[1] == self.num_heads

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        if mask is not None:
            attn = attn * mask_trans_att

        x = (attn @ v).transpose(1, 2)
        x = x.reshape(B, N, C)
        x = self.proj(x)
        return x


class ConvFormer_MTL(nn.Module):
    def __init__(self):
        super(ConvFormer_MTL, self).__init__()
        
        self.seg_branch = UNet_ConvFormer(pretrained=False, size='base')
        self.cls_branch = Classifier(embed_dim=576, num_classes=2)

    def forward(self, x, mask=None):
        
        logits, x_t = self.seg_branch(x)
        
        if mask is not None:
            out_cls = self.cls_branch(x_t, mask) 
        else: 
            out_cls = self.cls_branch(x_t, logits)    
        out_cls = torch.softmax(out_cls, dim=1)   

        return logits, out_cls



#
# BUS_DiagnosisTest
#

class BUS_DiagnosisTest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_BUS_Diagnosis1()

  def test_BUS_Diagnosis1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """
    pass

