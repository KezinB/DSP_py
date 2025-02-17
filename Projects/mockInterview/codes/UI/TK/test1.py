import tkinter as tk
from tkinter import filedialog, Label, Button, Text
import cv2
from PIL import Image, ImageTk
import threading

class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Interview Preparation App")
        self.root.geometry("800x600")
        self.root.configure(bg='#FFFFFF')  # Background color

        font = ("Verdana", 12)  # Font setting

        self.name_label = Label(root, text="Name:", font=font, bg='#F0F0F0')
        self.name_label.pack()
        self.name_entry = Text(root, height=1, width=30, font=font)
        self.name_entry.pack()

        self.email_label = Label(root, text="Email:", font=font, bg='#F0F0F0')
        self.email_label.pack()
        self.email_entry = Text(root, height=1, width=30, font=font)
        self.email_entry.pack()

        self.upload_button = Button(root, text="Upload Resume", command=self.upload_resume, font=font)
        self.upload_button.pack()

        self.resume_label = Label(root, text="", font=font, bg='#F0F0F0')
        self.resume_label.pack()

        self.start_interview_button = Button(root, text="Start Interview", command=self.start_interview, font=font)
        self.start_interview_button.pack()

        self.video_label = Label(root, bg='#F0F0F0')
        self.video_label.pack()

        self.question_label = Label(root, text="", font=font, bg='#F0F0F0')
        self.question_label.pack()

        self.cap = None
        self.video_thread = None

    def upload_resume(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        self.resume_label.config(text="Resume uploaded: " + file_path)
        # Here you can add code to parse the resume and generate questions

    def start_interview(self):
        self.cap = cv2.VideoCapture(0)
        self.question_label.config(text="Question: Tell me about yourself.")
        self.video_thread = threading.Thread(target=self.video_stream)
        self.video_thread.start()
        
    def video_stream(self):
        while True:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (320, 240))  # Reduce camera preview dimensions
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
            else:
                break
        
    def stop_interview(self):
        if self.cap:
            self.cap.release()
            self.cap = None

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
