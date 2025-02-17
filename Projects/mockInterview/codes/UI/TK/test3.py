import tkinter as tk
from tkinter import filedialog, ttk
import cv2
from PIL import Image, ImageTk
import threading


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Interview Preparation App")
        self.root.geometry("800x600")  # Set initial window size
        self.root.configure(bg="#22577E")

        # Configure grid for centering content
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Main frame for widgets
        self.main_frame = tk.Frame(self.root, bg="#95D1CC")
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Configure grid for centering inside main_frame
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(6, weight=1)
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_columnconfigure(2, weight=1)

        # Styling
        self.font = ("Verdana", 12)
        self.bg_color = "#95D1CC"
        self.accent_color = "#5584AC"
        self.text_color = "#22577E"
        self.style_buttons()


        # Header
        self.header_label = tk.Label(
            self.main_frame,
            text="Interview Preparation App",
            font=("Verdana", 16, "bold"),
            bg=self.accent_color,
            fg="#ffffff",
            pady=10,
        )
        self.header_label.grid(row=0, column=1, pady=20)

        # Name Entry with placeholder
        self.name_entry = ttk.Entry(self.main_frame, width=30, font=self.font)
        self.name_entry.grid(row=1, column=1, padx=20, pady=10, sticky="w")
        self.name_entry.insert(0, "Name")  # Add placeholder text
        self.name_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(self.name_entry, "Name"))
        self.name_entry.bind("<FocusOut>", lambda event: self.add_placeholder(self.name_entry, "Name"))

        # Email Entry with placeholder
        self.email_entry = ttk.Entry(self.main_frame, width=30, font=self.font)
        self.email_entry.grid(row=2, column=1, padx=20, pady=10, sticky="w")
        self.email_entry.insert(0, "Email")  # Add placeholder text
        self.email_entry.bind("<FocusIn>", lambda event: self.clear_placeholder(self.email_entry, "Email"))
        self.email_entry.bind("<FocusOut>", lambda event: self.add_placeholder(self.email_entry, "Email"))


        # Resume Upload
        self.upload_button = ttk.Button(self.main_frame, text="Upload Resume", command=self.upload_resume,style="Rounded.TButton")
        self.upload_button.grid(row=3, column=0, columnspan=3, pady=10)

        self.resume_label = tk.Label(self.main_frame, text="", font=self.font, bg=self.bg_color, fg=self.text_color)
        self.resume_label.grid(row=4, column=0, columnspan=3, pady=10)

        # Start Interview Button
        self.start_interview_button = ttk.Button(self.main_frame, text="Start Interview", command=self.show_interview_ui,style="Rounded.TButton")
        self.start_interview_button.grid(row=5, column=0, columnspan=3, pady=20)

        # Video UI elements
        self.video_label = None
        self.cap = None
        self.video_thread = None

    def upload_resume(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.resume_label.config(text=f"Resume uploaded: {file_path.split('/')[-1]}")

    def show_interview_ui(self):
        # Clear main UI and show interview-related elements
        for widget in self.main_frame.winfo_children():
            widget.destroy()

        # Layout Configuration
        self.main_frame.grid_rowconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=6)
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Top Section (Video Controls and Back Button)
        top_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        top_frame.grid(row=0, column=0, sticky="nsew")
        top_frame.grid_columnconfigure(0, weight=1)

        # Back Button
        back_button = ttk.Button(self.main_frame, text="Back", command=self.show_main_ui, style="Rounded.TButton")
        back_button.grid(row=0, column=0, sticky="w", padx=20, pady=10)

        # Video Controls Button (Start, End Video)
        start_video_button = ttk.Button(self.main_frame, text="Start Video", command=self.start_video, style="Rounded.TButton")
        start_video_button.grid(row=2, column=0, pady=10)

        end_video_button = ttk.Button(self.main_frame, text="End Video", command=self.end_video, style="Rounded.TButton")
        end_video_button.grid(row=2, column=1, pady=10)

        # Middle Section (Video Feed)
        video_frame = tk.Frame(self.main_frame, bg="#f9f9f9")
        video_frame.grid(row=1, column=0, sticky="nsew")
        video_frame.grid_rowconfigure(0, weight=1)
        video_frame.grid_columnconfigure(0, weight=1)

        self.video_label = tk.Label(video_frame, bg="#000000")
        self.video_label.grid(row=0, column=0, padx=10, pady=10)

        # Bottom Section (Questions)
        question_frame = tk.Frame(self.main_frame, bg=self.bg_color)
        question_frame.grid(row=2, column=0, sticky="nsew")
        question_frame.grid_rowconfigure(0, weight=1)
        question_frame.grid_columnconfigure(0, weight=1)

        question_label = tk.Label(
            question_frame, text="Question: Tell me about yourself.", font=self.font, bg=self.bg_color, fg=self.text_color
        )
        question_label.grid(row=0, column=0, pady=20)


    def show_main_ui(self):
        # Stop video feed and reset UI to the main page
        self.end_video()
        for widget in self.main_frame.winfo_children():
            widget.destroy()
        self.__init__(self.root)

    def start_video(self):
        if not self.cap:
            self.cap = cv2.VideoCapture(0)
        
        # Check if camera opened successfully
        if not self.cap.isOpened():
            print("Error: Could not open video camera.")
            return  # Exit if camera couldn't be opened
        
        self.video_thread = threading.Thread(target=self.video_stream, daemon=True)
        self.video_thread.start()

    def video_stream(self):
        while self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (320, 240))  # Resize frame to fit the label
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_label.imgtk = imgtk
                self.video_label.config(image=imgtk)
                self.root.update_idletasks()  # Ensure the window updates correctly
            else:
                break


    def end_video(self):
        if self.cap:
            self.cap.release()
            self.cap = None
            self.video_label.config(image="")  # Clear video feed
            
    def clear_placeholder(self, entry, placeholder):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(foreground="black")  # Set text color to black when focused

    def add_placeholder(self, entry, placeholder):
        if not entry.get():
            entry.insert(0, placeholder)
            entry.config(foreground="gray")  # Set text color to gray for placeholder
            
    def style_buttons(self):
        style = ttk.Style()
        style.configure("Rounded.TButton",
                    font=("Verdana", 14),  # Increase font size
                    padding=(5, 10),  # Increase padding to make the button larger
                    relief="flat",
                    background="#5584AC",
                    foreground="Black",)
        style.map("Rounded.TButton",
              background=[('active', '#22577E')],
              relief=[('pressed', 'sunken')],
              foreground=[('active', 'Black')],
              )

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
