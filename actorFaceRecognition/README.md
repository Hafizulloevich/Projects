# 🎬 Actor Face Recognition  

An interactive machine learning project that recognizes famous actors’ faces using **Support Vector Machines (SVM)** and **Wavelet Transforms**.  

🔗 **Live Demo on Hugging Face Spaces**: [Try it here!](https://huggingface.co/spaces/hafizulloevich/actorFaceRecognition)  

---

## 📌 Project Overview  

This project aims to build an actor face recognition system using traditional machine learning techniques instead of deep learning.  
The pipeline includes:  

1. **Web Scraping** → Collected images of famous actors.  
2. **Preprocessing** →  
   - Removed irrelevant images.  
   - Detected faces and cropped them using Haar Cascade.  
   - Applied **Wavelet Transform** for feature extraction.  
3. **Model Training** →  
   - Combined raw pixel values with wavelet features.  
   - Built a pipeline with **StandardScaler** and **SVM (RBF kernel, C=10)**.  
   - Achieved **~95% accuracy** on test data.  
4. **Deployment** →  
   - Created an interactive web app using **Gradio**.  
   - Hosted on **Hugging Face Spaces**.  

---

## 🗂 Dataset  

- Images were collected via **web scraping** using [`image_webscraper.ipynb`](.image_web_scrapping/image_webscraper.ipynb).  
- Dataset contains cropped face images of:  
  - Angelina Jolie  
  - Denzel Washington  
  - Jackie Chan  
  - Jason Statham  
  - Leonardo DiCaprio  
  - Tom Cruise
- Data preprocessing [`notebook`]('actors_image_classification.ipynb').
- [`Model`]('saved_model.pkl) for using.
- dataset -  is the folder where the images were downloaded
- dataset/cropped - is the folder where the processed photos went.
- test_images - sample photos for checking the model performance
---

## 🛠️ Tech Stack  

- **Python**  
- **OpenCV** – Face & eye detection  
- **PyWavelets** – Wavelet transforms  
- **Scikit-learn** – StandardScaler & SVM  
- **Gradio** – Interactive UI for deployment  
- **Hugging Face Spaces** – Hosting  

---

## 🚀 How to Run Locally  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/actorFaceRecognition.git
   cd actorFaceRecognition
or 

Just install all the files and run the requirements.txt to be able to import the libraries
