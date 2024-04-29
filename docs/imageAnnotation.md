# Image Annotation Process

## Overview

This section outlines the process of annotating images using the DagsHub integration with Label Studio, a popular tool for data labeling tasks. The annotation process plays a crucial role in preparing data for training machine learning models by classifying images into predefined categories.

## Setting Up Label Studio with DagsHub

### Integration Overview

Label Studio is integrated with DagsHub to streamline the annotation workflow and facilitate the management and storage of annotated datasets. This integration allows for seamless synchronization of annotation data with DagsHub repositories, making it easier to track and version annotation progress.

### Initial Setup

To start annotating images using Label Studio within a DagsHub environment:

1. **Connect Your DagsHub Repository**:
   - Ensure that your DagsHub repository is set up to store and manage your annotation data.
   - Configure the repository settings in Label Studio to point to your specific DagsHub repository.

2. **Configure Label Studio**:
   - Launch Label Studio and create a new project.
   - Choose the type of task (image classification, object detection, etc.) and define the label classes.

### Configuring Annotation Classes

For the task of image classification, define the following six classes which correspond to the categories required for your model:

- Class 1
- Class 2
- Class 3
- Class 4
- Class 5
- Class 6

Each class should be clearly described to ensure that annotators understand the criteria for each classification.

## Annotation Process

### Annotating Images

1. **Upload Images**:
   - Images to be annotated are uploaded to the Label Studio project directly or synchronized from a designated folder in your DagsHub repository.

2. **Manual Annotation**:
   - Annotators classify each image by assigning one of the six predefined classes.
   - Ensure each image is viewed and classified accurately to maintain high-quality data for model training.

### Quality Assurance

- Regular checks and reviews should be conducted to maintain the consistency and accuracy of the annotations.
- Discrepancies or uncertainties in image classification should be discussed and resolved to refine the annotation guidelines.

## Exporting Annotations

Once the annotation process is complete, the annotated data can be exported from Label Studio into a CSV format, which is convenient for further processing and analysis in data preparation steps.

### Export Procedure

- Navigate to the export section in Label Studio.
- Select the CSV format for export.
- Download the annotated dataset, which will include image file references and their corresponding class labels.

## Conclusion

The integration of Label Studio with DagsHub facilitates a robust framework for annotating images, ensuring that the data used for training machine learning models is accurately classified and easily accessible. This documentation provides a clear guide on how to manage the image annotation process effectively within this integrated environment.

