import tkinter as tk


def visualize_bst(root):
    window = tk.Tk()
    window.title("BST Visualizer")

    canvas = tk.Canvas(window, width=1200, height=800)
    canvas.pack(fill="both", expand=True)

    def draw_node(node, x, y, spacing):
        if node is None:
            return

        radius = 20

        if node.left:
            child_x = x - spacing
            child_y = y + 80

            canvas.create_line(x, y, child_x, child_y)
            draw_node(node.left, child_x, child_y, spacing / 2)

        if node.right:
            child_x = x + spacing
            child_y = y + 80

            canvas.create_line(x, y, child_x, child_y)
            draw_node(node.right, child_x, child_y, spacing / 2)

        canvas.create_oval(
            x - radius,
            y - radius,
            x + radius,
            y + radius
        )

        canvas.create_text(
            x,
            y,
            text=str(node.val)
        )

    draw_node(root, 600, 50, 300)

    window.mainloop()