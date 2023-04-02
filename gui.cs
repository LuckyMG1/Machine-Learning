using System;
using System.Drawing;
using System.Windows.Forms;

public class Program {
  public static void Main(string[] args) {
    Application.Run(new UI());
  }
}

public class UI : Form {
  private Rectangle screen = Screen.PrimaryScreen.Bounds;

  private void AddButtons() {
    var create_network_Button = new Button();
    var start_training_Button = new Button();

    create_network_Button.Text = "Create new network";
    create_network_Button.Location = new Point(300,200);
    create_network_Button.Size = new Size(200,100);

    start_training_Button.Text = "Start training";
    start_training_Button.Location = new Point(600,200);
    start_training_Button.Size = new Size(200,100);

    this.Controls.Add(create_network_Button);
    this.Controls.Add(start_training_Button);
  }
  public UI() {
    this.Size = new Size(screen.Width, screen.Height);
    AddButtons();
  }
}
