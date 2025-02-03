import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import threading


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Interview Preparation App")
        self.root.attributes('-fullscreen', True)  # Full-screen mode
        self.root.configure(bg="#f9f9f9")

        # Configure grid to center content
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Main frame to hold all widgets
        self.main_frame = tk.Frame(self.root, bg="#f9f9f9")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Styling
        self.font = ("Arial", 12)
        self.bg_color = "#ffffff"
        self.accent_color = "#4CAF50"
        self.text_color = "#333333"

        # Header
        self.header_label = tk.Label(
            self.main_frame,
            text="Interview Preparation App",
            font=("Arial", 16, "bold"),
            bg=self.accent_color,
            fg="#ffffff",
            pady=10,
        )
        self.header_label.grid(row=0, column=0, columnspan=2, sticky="ew", padx=20, pady=20)

        # Name
        self.name_label = tk.Label(self.main_frame, text="Name:", font=self.font, bg=self.bg_color, fg=self.text_color)
        self.name_label.grid(row=1, column=0, padx=20, pady=10, sticky="e")
        self.name_entry = ttk.Entry(self.main_frame, width=30, font=self.font)
        self.name_entry.grid(row=1, column=1, padx=20, pady=10, sticky="w")

        # Email
        self.email_label = tk.Label(self.main_frame, text="Email:", font=self.font, bg=self.bg_color, fg=self.text_color)
        self.email_label.grid(row=2, column=0, padx=20, pady=10, sticky="e")
        self.email_entry = ttk.Entry(self.main_frame, width=30, font=self.font)
        self.email_entry.grid(row=2, column=1, padx=20, pady=10, sticky="w")

        # Resume Upload
        self.upload_button = ttk.Button(self.main_frame, text="Upload Resume", command=self.upload_resume)
        self.upload_button.grid(row=3, column=0, columnspan=2, pady=10)

        self.resume_label = tk.Label(self.main_frame, text="", font=self.font, bg=self.bg_color, fg=self.text_color)
        self.resume_label.grid(row=4, column=0, columnspan=2, pady=10)

        # Start Interview Button
        self.start_interview_button = ttk.Button(self.main_frame, text="Start Interview", command=self.start_interview)
        self.start_interview_button.grid(row=5, column=0, columnspan=2, pady=20)

        # Video Feed
        self.video_label = tk.Label(self.main_frame, bg=self.bg_color)
        self.video_label.grid(row=6, column=0, columnspan=2, pady=10)

        # Question
        self.question_label = tk.Label(self.main_frame, text="", font=self.font, bg=self.bg_color, fg=self.text_color)
        self.question_label.grid(row=7, column=0, columnspan=2, pady=10)

        # Variables for video and threading
        self.cap = None
        self.video_thread = None

        # Escape key to exit full-screen
        self.root.bind("<Escape>", self.exit_fullscreen)

    def upload_resume(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.resume_label.config(text=f"Resume uploaded: {file_path.split('/')[-1]}")

    def start_interview(self):
        self.cap = cv2.VideoCapture(0)
        self.question_label.config(text="Question: Tell me about yourself.")
        self.video_thread = threading.Thread(target=self.video_stream, daemon=True)
        self.video_thread.start()

    def video_stream(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (320, 240))
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

    def exit_fullscreen(self, event=None):
        self.root.attributes('-fullscreen', False)


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
