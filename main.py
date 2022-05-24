import tkinter as tk

from app import MainApplication

if __name__ == "__main__":
    root = tk.Tk()

    width = 1000
    height = 800

    positionRight = int(root.winfo_screenwidth() / 2 - width / 2)
    positionDown = int(root.winfo_screenheight() / 2 - height / 2)

    is_model_pretrained = False
    MainApplication(is_model_pretrained, root).pack(side="top", fill="both", expand=True)

    root.title("HumusER")
    root.geometry("+{}+{}".format(positionRight, positionDown))
    root.mainloop()

