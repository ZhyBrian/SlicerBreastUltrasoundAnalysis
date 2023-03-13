import logging
import os

import vtk

import slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import numpy as np
import math
import qt
import copy

try:
  import cv2
  from PIL import Image
  import torch
  import torch.nn as nn
  import torch.utils.model_zoo as model_zoo
  from torchvision import transforms
except:
  slicer.util.pip_install('torch torchvision torchaudio')
  slicer.util.pip_install('opencv-python')
  import cv2
  from PIL import Image
  import torch
  import torch.nn as nn
  import torch.utils.model_zoo as model_zoo
  from torchvision import transforms
finally:
  logging.info(f'From BUS_Diagnosis: torch version: {torch.__version__}')
  logging.info(f'From BUS_Diagnosis: cv2 version: {cv2.__version__}')

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
    self.parent.contributors = ["Yi Zhang (Chongqing University, China), Xiaojun Chen (Shanghai Jiao Tong University, China)"]  # TODO: replace with "Firstname Lastname (Organization)"
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
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI1)
    self.ui.outputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI2)
    self.ui.segmentAllCheckBox.connect("toggled(bool)", self.updateParameterNodeFromGUI3)

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
    self.progressDiag.setFixedSize(330, 50)
    self.progressDiag.setAutoClose(False)
    self.progressDiag.setCancelButton(None)
    self.progressDiag.close()
    
    self.saveCount = 0
    
    # Buttons
    self.ui.applyButton.connect('clicked(bool)', self.onApplyButton)
    self.ui.installPytorchButton.connect('clicked(bool)', self.onInstallPytorchButton)
    self.ui.installcv2Button.connect('clicked(bool)', self.onInstallcv2Button)
    self.ui.pushButtonHideSeg.connect('clicked(bool)', self.onPushButtonHideSeg)
    self.ui.pushButtonShowSeg.connect('clicked(bool)', self.onPushButtonShowSeg)
    self.ui.movetoOffsetButton.connect('clicked(bool)', self.onMovetoOffsetButton)
    self.ui.pushButtonSaveResults.connect('clicked(bool)', self.onPushButtonSaveResults)

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
    if self._parameterNode is not None:
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
    # Update node selectors and sliders
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))
    self.ui.outputSelector.setCurrentNode(self._parameterNode.GetNodeReference("OutputVolume"))
    self.ui.segmentAllCheckBox.checked = (self._parameterNode.GetParameter("SegmentAll") == "true")

    # Update buttons states and tooltips
    self.ui.installPytorchButton.toolTip = "Install up-to-date Pytorch(cpu version) to enable this module"
    self.ui.installcv2Button.toolTip = "Install up-to-date opencv-python to enable this module"
    if self._parameterNode.GetNodeReference("InputVolume") and self._parameterNode.GetNodeReference("OutputVolume"):
      if self.ui.segmentAllCheckBox.checked:
        self.ui.applyButton.toolTip = "Predict segmentation results for all slices"
      else:
        self.ui.applyButton.toolTip = "Predict segmentation and diagnosis results only for the current slice"
      self.ui.applyButton.enabled = True
    else:
      self.ui.applyButton.toolTip = "Select input and output volume nodes first"
      self.ui.applyButton.enabled = False
      
    self.UpdateShowHideButtonStatus()
    self.UpdateMovetoOffsetButtonStatus()
    self.UpdatePushButtonSaveResultsStatus()

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False
    
    
  def updateParameterNodeFromGUI1(self, caller=None, event=None):
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

  def updateParameterNodeFromGUI2(self, caller=None, event=None):
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
  
  def updateParameterNodeFromGUI3(self, caller=None, event=None):
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
      
      self.progressDiag.show()
      # Compute output
      isSegmentMoreThanOneSlice = self.logic.process(self.ui.inputSelector.currentNode(), 
          self.ui.outputSelector.currentNode(), self.progressDiag, self.ui.segmentAllCheckBox.checked)
      
      if isSegmentMoreThanOneSlice == -1:
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
  
  def onInstallcv2Button(self):
    slicer.util.pip_install('opencv-python')
    import cv2
    logging.info(f'From BUS_Diagnosis: cv2 version: {cv2.__version__}')
  
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
      segmentId = segmentNode.GetSegmentation().GetSegmentIdBySegmentName(f'Segment_1')
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
      
      cv2.imwrite(os.path.join(savePath, f"Image.png"), inputVolumeArray)
      cv2.imwrite(os.path.join(savePath, f"Label_AIpredicted.png"), np.flip(np.flip(outputVolumeArray, axis=0), axis=1))
      cv2.imwrite(os.path.join(savePath, f"Label_Revised.png"), segmentArray*255)
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
    self.model_seg = Unet(num_seg=1, num_class=2, pretrained=False, backbone='resnet50').to(self.device).eval()
    weight_path_seg = os.path.join(os.path.dirname(__file__), 'Resources/net_weight.pth')
    if os.path.exists(weight_path_seg):
      self.model_seg.load_state_dict(torch.load(weight_path_seg, map_location=torch.device(self.device)))
      logging.info('Net successfully loaded weight!')
    else:
      logging.info('Net weight failed to be loaded!')
      qt.QMessageBox.information(slicer.util.mainWindow(), "Warning", "Net weight failed to be loaded. \nAI is unable to provide appropriate diagnosis results.")

    self.transform = transforms.Compose([transforms.ToTensor()]) 
    self.wh = 384
    self.segmentationNode = None
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

  def keep_image_size_open_mask_test(self, input_img, size, scale_method='ANTIALIAS'):
    logging.info(f'input image size: {input_img.shape}')
    img = Image.fromarray(np.float32(input_img))

    temp = max(img.size)
    mask = Image.new('L', (temp, temp))
    mask.paste(img, (0, 0))
    if scale_method == 'ANTIALIAS':
        mask = mask.resize(size, Image.ANTIALIAS) 
    else:
        mask = mask.resize(size, Image.NEAREST) 
    return mask, img.size[0], img.size[1]
  
  def fill_binary_fig_holes_and_eliminate_noise(self, blank_thresh):
    blank_thresh = cv2.medianBlur(blank_thresh, 3)
    
    blank_thresh_copy = copy.deepcopy(blank_thresh)
    h, w = blank_thresh_copy.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(blank_thresh_copy, mask, (0,0), 255)
    blank_thresh_copy_inv = cv2.bitwise_not(blank_thresh_copy)
    blank_thresh_out = blank_thresh | blank_thresh_copy_inv
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # rectangular
    blank_thresh_out = cv2.morphologyEx(blank_thresh_out, cv2.MORPH_OPEN, kernel, iterations=1)
    blank_thresh_out = cv2.medianBlur(blank_thresh_out, 3)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blank_thresh_out, connectivity=4)
    if num_labels <= 2:
      pass
    else:
      areas = stats[1:, 4]
      maxArea = np.max(areas)
      for i in range(1, num_labels):
        currentArea = stats[i, 4]
        if currentArea < 0.1 * maxArea:
          labels[labels == i] = 0
      blank_thresh_out = blank_thresh_out * labels
      blank_thresh_out[blank_thresh_out > 255] = 255
    
    return blank_thresh_out
  
  def AIAssistedSegment(self, input_img):
    input_img, w, h = self.keep_image_size_open_mask_test(input_img, size=(self.wh, self.wh))
    input_img = torch.cat([self.transform(input_img), self.transform(input_img), self.transform(input_img)], dim = 0)
    input_img = torch.unsqueeze(input_img, dim=0).to(self.device)
    result_img, result_cls = self.model_seg(input_img)
    result_img = transforms.ToPILImage()(result_img[0].float())
    result_img = result_img.crop(box=(0, 0, w/(max(w,h)/self.wh), h/(max(w,h)/self.wh)))  
    result_img = result_img.resize((w, h), Image.ANTIALIAS)
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
 
    
  def process(self, inputVolume, outputVolume, progressDiag, SegmentAll=False):
    """
    Run the processing algorithm.
    Can be used without GUI widget.
    :param inputVolume: volume to be segmented and classified
    :param outputVolume: segmentation result
    :param SegmentAll: if True then all the slice will be segmented simultaneously, otherwise only current slice will be segmented
    """  

    if not inputVolume or not outputVolume:
      raise ValueError("Input or output volume is invalid")

    import time
    startTime = time.time()
    logging.info('Processing started')
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
      logging.info('input should be 3D volume!')
      return -1
    
    h, w = input_img.shape[0], input_img.shape[1]
    progressDiag.setMaximum(self.numberOfChannels)
    progressDiag.setValue(0)
    
    if self.isMultipleChannel:
      result_img_temp = np.zeros((h, w, self.numberOfChannels))
      if len(input_img.shape) >= 3:
        assert self.segmentAll == True
        for i in range(self.numberOfChannels):
          result_img_temp[:, :, i] = np.squeeze(self.AIAssistedSegment(input_img[:, :, i])[0], axis=2)
          progressDiag.setValue(i)
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

    slicer.mrmlScene.RemoveNode(self.segmentationNode)
    self.segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    self.segmentationNode.CreateDefaultDisplayNodes()
    self.segmentationNode.GetSegmentation().AddEmptySegment()
    segmentId = self.segmentationNode.GetSegmentation().GetSegmentIdBySegmentName(f'Segment_1')
    slicer.util.updateSegmentBinaryLabelmapFromArray(result_img, self.segmentationNode, segmentId, inputVolume)

    slicer.util.updateVolumeFromArray(outputVolume, np.flip(np.flip(result_img, axis=1), axis=2))
    progressDiag.setValue(self.numberOfChannels)

    stopTime = time.time()
    logging.info(f'Processing completed in {stopTime-startTime:.2f} seconds')
    
    if self.isMultipleChannel and self.segmentAll:
      return 1   # isSegmentMoreThanOneSlice = 1
    else:
      return 0   # isSegmentMoreThanOneSlice = 0




def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
  return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
  expansion = 1

  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
    super(BasicBlock, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    if groups != 1 or base_width != 64:
      raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
      raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    self.conv1 = conv3x3(inplanes, planes, stride)
    self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = conv3x3(planes, planes)
    self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class Bottleneck(nn.Module):
  expansion = 4
  def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
    super(Bottleneck, self).__init__()
    if norm_layer is None:
      norm_layer = nn.BatchNorm2d
    width = int(planes * (base_width / 64.)) * groups
    self.conv1 = conv1x1(inplanes, width)
    self.bn1 = norm_layer(width)
    self.conv2 = conv3x3(width, width, stride, groups, dilation)
    self.bn2 = norm_layer(width)
    self.conv3 = conv1x1(width, planes * self.expansion)
    self.bn3 = norm_layer(planes * self.expansion)

    self.relu = nn.ReLU(inplace=True)
    self.downsample = downsample
    self.stride = stride

  def forward(self, x):
    identity = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu(out)

    out = self.conv3(out)
    out = self.bn3(out)

    if self.downsample is not None:
      identity = self.downsample(x)

    out += identity
    out = self.relu(out)

    return out


class ResNet(nn.Module):
  def __init__(self, block, layers, num_classes=1000):
    self.inplanes = 64
    super(ResNet, self).__init__()
    self.conv1  = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1    = nn.BatchNorm2d(64)
    self.relu   = nn.ReLU(inplace=True)
    self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
    self.layer1 = self._make_layer(block, 64, layers[0])
    self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
    self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
    self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
    
    self.avgpool = nn.AvgPool2d(7)
    self.fc = nn.Linear(512 * block.expansion, num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

  def _make_layer(self, block, planes, blocks, stride=1):
    downsample = None
    if stride != 1 or self.inplanes != planes * block.expansion:
      downsample = nn.Sequential(
            nn.Conv2d(self.inplanes, planes * block.expansion,
                kernel_size=1, stride=stride, bias=False),
        nn.BatchNorm2d(planes * block.expansion),
    )

    layers = []
    layers.append(block(self.inplanes, planes, stride, downsample))
    self.inplanes = planes * block.expansion
    for i in range(1, blocks):
      layers.append(block(self.inplanes, planes))

    return nn.Sequential(*layers)

  def forward(self, x):
    x       = self.conv1(x)
    x       = self.bn1(x)
    feat1   = self.relu(x)

    x       = self.maxpool(feat1)
    feat2   = self.layer1(x)

    feat3   = self.layer2(feat2)
    feat4   = self.layer3(feat3)
    feat5   = self.layer4(feat4)
    return [feat1, feat2, feat3, feat4, feat5]

def resnet50(pretrained=False, **kwargs):
  model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
  if pretrained:
    model.load_state_dict(model_zoo.load_url('https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth', model_dir='model_data'), strict=False)
  
  del model.avgpool
  del model.fc
  return model


class unetUp(nn.Module):
  def __init__(self, in_size, out_size):
    super(unetUp, self).__init__()
    self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
    self.bn1    = nn.BatchNorm2d(out_size)
    self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
    self.bn2    = nn.BatchNorm2d(out_size)
    self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
    self.relu   = nn.ReLU(inplace = True)

  def forward(self, inputs1, inputs2):
    outputs = torch.cat([inputs1, self.up(inputs2)], 1)
    outputs = self.conv1(outputs)
    outputs = self.bn1(outputs)
    outputs = self.relu(outputs)
    outputs = self.conv2(outputs)
    outputs = self.bn2(outputs)
    outputs = self.relu(outputs)
    return outputs

class BasicLinearBlock(nn.Module):
  def __init__(self, in_neuron, out_neuron, dropoutrate=0.0):
    super(BasicLinearBlock, self).__init__()
    self.fc = nn.Linear(in_neuron, out_neuron)
    self.bn1 = nn.BatchNorm1d(out_neuron)
    self.activate = nn.ReLU()
    self.drop = nn.Dropout(dropoutrate)
  
  def forward(self, x):
    x = self.fc(x)
    x = self.bn1(x)
    x = self.activate(x)
    x = self.drop(x)
    return x

class Unet(nn.Module):
  def __init__(self, num_seg = 1, num_class = 2, pretrained = False, backbone = 'resnet50'):
    super(Unet, self).__init__()
    if backbone == 'vgg':
      pass
    elif backbone == "resnet50":
      self.resnet = resnet50(pretrained = pretrained)
      in_filters  = [192, 512, 1024, 3072]
    else:
      raise ValueError('Unsupported backbone - `{}`, Use vgg, resnet50.'.format(backbone))
    out_filters = [64, 128, 256, 512]

    self.up_concat4 = unetUp(in_filters[3], out_filters[3])
    self.up_concat3 = unetUp(in_filters[2], out_filters[2])
    self.up_concat2 = unetUp(in_filters[1], out_filters[1])
    self.up_concat1 = unetUp(in_filters[0], out_filters[0])

    if backbone == 'resnet50':
      self.up_conv = nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor = 2), 
        nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_filters[0]),
        nn.ReLU(),
        nn.Conv2d(out_filters[0], out_filters[0], kernel_size = 3, padding = 1),
        nn.BatchNorm2d(out_filters[0]),
        nn.ReLU(),
      )
    else:
      self.up_conv = None

    self.final = nn.Conv2d(out_filters[0], num_seg, 1)
    self.output = nn.Sigmoid()

    self.backbone = backbone
    
    self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    self.fc = nn.Sequential(
      BasicLinearBlock(in_filters[3] + out_filters[3], 1024),
      BasicLinearBlock(1024, 256),
      nn.Linear(256, num_class)
    )

  def forward(self, inputs):
    if self.backbone == "vgg":
      pass
    elif self.backbone == "resnet50":
      [feat1, feat2, feat3, feat4, feat5] = self.resnet.forward(inputs)

    up4 = self.up_concat4(feat4, feat5)
    up3 = self.up_concat3(feat3, up4)
    up2 = self.up_concat2(feat2, up3)
    up1 = self.up_concat1(feat1, up2)

    if self.up_conv != None:
      up1 = self.up_conv(up1)

    out = self.final(up1)
    out_img = self.output(out)
    
    feat4_n = torch.flatten(self.avgpool(feat4), 1)
    feat4_n = feat4_n / torch.sqrt(torch.pow(feat4_n, 2).sum(dim=1).unsqueeze(dim=1))
    feat5_n = torch.flatten(self.avgpool(feat5), 1)
    feat5_n = feat5_n / torch.sqrt(torch.pow(feat5_n, 2).sum(dim=1).unsqueeze(dim=1))
    up4_n = torch.flatten(self.avgpool(up4), 1)
    up4_n = up4_n / torch.sqrt(torch.pow(up4_n, 2).sum(dim=1).unsqueeze(dim=1))
    cls_br = torch.cat([feat4_n, feat5_n, up4_n], 1)
    out_cls = self.fc(cls_br)
    out_cls = torch.softmax(out_cls, dim=1)
    
    return out_img, out_cls



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

