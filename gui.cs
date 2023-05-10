using System;
using System.IO;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Windows.Forms;
using System.Collections.Generic;
using NeuralNetwork;

public class Program {
  public static void Main(string[] args) {
    Application.Run(new UI());
  }
}

public class UI : Form {
  private Rectangle screen = Screen.PrimaryScreen.Bounds;

  private PictureBox canvas;
  private bool mouse_down;
  private Point last_point = Point.Empty;

  private InputLayer networkInputLayer;
  private OutputLayer networkOutputLayer;

  // t stands for training
  private float t_learning_rate;
  private float t_relu_coefficient;
  private List<int> t_neuron_config;

  private void InitialiseNeuralNetwork() {
    networkInputLayer = new InputLayer(784,512);
    networkOutputLayer = new OutputLayer();

    networkInputLayer.AddLayer(new Layer(512,256));
    networkInputLayer.AddLayer(new Layer(256,10));
    networkInputLayer.AddLayer(networkOutputLayer);
  }

  private void InitialiseTabs() {
    // Create tab control
    TabControl tab_control = new TabControl();
    tab_control.Dock = DockStyle.Fill;

    // Create tabs
    TabPage tab_page1 = new TabPage("Use trained model");
    TabPage tab_page2 = new TabPage("Train model");

    // Set up tabs
    tab_page1 = Tab1Components(tab_page1);
    tab_page2 = Tab2Components(tab_page2);

    // add tabs to control
    tab_control.TabPages.Add(tab_page1);
    tab_control.TabPages.Add(tab_page2);

    Controls.Add(tab_control);
  }

  private TabPage Tab1Components(TabPage _tab_page) {
    // create label above canvas for clarity
    Label label = new Label();
    label.Text = "Draw your number here:";
    label.Location = new Point(350,80);
    _tab_page.Controls.Add(label);

    // create a button to detect the number drawn in the canvas
    Button button_detect = new Button();
    button_detect.Text = "Detect Number";
    button_detect.Location = new Point(850,150);
    button_detect.Size = new Size(150,90);
    button_detect.FlatStyle = FlatStyle.Flat;
    button_detect.BackColor = Color.LightGray;
    _tab_page.Controls.Add(button_detect);

    // create a button to clear the canvas
    Button button_clear = new Button();
    button_clear.Text = "Clear Canvas";
    button_clear.Location = new Point(850,300);
    button_clear.Size = new Size(150,90);
    button_clear.FlatStyle = FlatStyle.Flat;
    button_clear.BackColor = Color.LightGray;
    _tab_page.Controls.Add(button_clear);

    // set up clear button click action
    button_clear.Click += delegate(object sende, EventArgs e) {
      canvas.Image = new Bitmap(canvas.Width, canvas.Height);
      Invalidate();
    };

    button_detect.Click += new EventHandler(button_detect_Click);

    _tab_page.Controls.Add(canvas);
    return _tab_page;
  }

  private TabPage Tab2Components(TabPage _tab_page) {
    // create button to start training
    Button button_start_training = new Button();
    button_start_training.Text = "Start Training";
    button_start_training.Location = new Point(800,150);
    button_start_training.Size = new Size(250,150);
    button_start_training.FlatStyle = FlatStyle.Flat;
    button_start_training.BackColor = Color.LightGray;
    _tab_page.Controls.Add(button_start_training);

    // display loss as label under start training button
    Label loss_label = new Label();
    loss_label.Text = "Loss: ";
    loss_label.Location = new Point(800,350);
    loss_label.Size = new Size(100,50);
    _tab_page.Controls.Add(loss_label);

    // text box for learning rate
    TextBox lr_textbox = new TextBox();
    lr_textbox.Text = "0.01";
    lr_textbox.Location = new Point(250,200);
    _tab_page.Controls.Add(lr_textbox);
    Label lr_label = new Label();
    lr_label.Text = "Learning rate:";
    lr_label.Location = new Point(250,180);
    _tab_page.Controls.Add(lr_label);

    // text box for relu coefficie nt
    TextBox rc_textbox = new TextBox();
    rc_textbox.Text = "0.2";
    rc_textbox.Location = new Point(250,300);
    _tab_page.Controls.Add(rc_textbox);
    Label rc_label = new Label();
    rc_label.Text = "ReLU coefficient:";
    rc_label.Location = new Point(250,280);
    _tab_page.Controls.Add(rc_label);

    // text box for neuron config
    TextBox nc_textbox = new TextBox();
    nc_textbox.Text = "512,256";
    nc_textbox.Location = new Point(250,400);
    _tab_page.Controls.Add(nc_textbox);
    Label nc_label = new Label();
    nc_label.Text = "Neuron configuration:";
    nc_label.Location = new Point(250,370);
    _tab_page.Controls.Add(nc_label);

    // add press key event to the 3 textboxes
    lr_textbox.KeyDown += delegate(object sender, KeyEventArgs e) {
      if(e.KeyCode == Keys.Enter) {
        this.t_learning_rate = float.Parse(lr_textbox.Text);
      }
    };

    rc_textbox.KeyDown += delegate(object sender, KeyEventArgs e) {
      if(e.KeyCode == Keys.Enter) {
        this.t_relu_coefficient = float.Parse(rc_textbox.Text);
      }
    };

    nc_textbox.KeyDown += delegate(object sender, KeyEventArgs e) {
      if(e.KeyCode == Keys.Enter) {
        List<int> temp_list = new List<int>();
        string temp_string = "";
        for(int i = 0; i < nc_textbox.Text.Length; i++) {
          if(nc_textbox.Text[i] != ',') { temp_string += nc_textbox.Text[i]; }
          else
          {
            temp_list.Add(Convert.ToInt32(temp_string));
            temp_string = "";
          }
          if(i == nc_textbox.Text.Length-1) {
            temp_list.Add(Convert.ToInt32(temp_string));
          }
        }
        this.t_neuron_config = temp_list;
      }
    };

    // add click event to start training button
    button_start_training.Click += new EventHandler(button_start_training_Click);

    // create changeable parameter textboxes
    return _tab_page;
  }

  private Bitmap ResizeImage(Image _image, int width, int height) {
    Bitmap output_image = new Bitmap(width, height, PixelFormat.Format32bppArgb);
    using (Graphics g = Graphics.FromImage(output_image)) {
      g.Clear(Color.White);
      g.DrawImage(canvas.Image, 0, 0, width, height);
    }
    return output_image;
  }

  private byte ToByte(Color c) {
    int average_colour = c.R + c.G + c.B;
    average_colour /= 3;
    return (byte)(255-average_colour);
  }

  private void button_detect_Click(object sender, EventArgs e) {
    // resize image
    Bitmap resized_image = ResizeImage(canvas.Image, 28, 28);
    // convert image to binary form;
    byte[][] byte_image = new byte[28][];
    for(int i = 0; i < 28; i++) {
      byte_image[i] = new byte[28];
      for(int j = 0; j < 28; j++) {
        byte_image[i][j] = ToByte(resized_image.GetPixel(j,i));
      }
    }
    // pass image to neural network
    DigitImage image = new DigitImage(byte_image,(byte)(10));
    image.ConvertBinary();
    networkInputLayer.LoadImage(image);
    networkInputLayer.FeedForward();

    //output prediction
    Console.WriteLine(networkOutputLayer.Max());
  }

  private void button_start_training_Click(object sender, EventArgs e) {
    InputLayer t_input_layer = new InputLayer(784,t_neuron_config[0]);
    OutputLayer t_output_layer = new OutputLayer();
    t_output_layer.LearningRate = t_learning_rate;

    Layer t_current_layer = t_input_layer;
    t_current_layer.relu_parameter = t_relu_coefficient;

    for(int i = 0; i < t_neuron_config.Count; i++) {
      if(i == t_neuron_config.Count-1) {
        t_input_layer.AddLayer(new Layer(t_neuron_config[i],10));
      } else {
        t_input_layer.AddLayer(new Layer(t_neuron_config[i], t_neuron_config[i+1]));
      }
      t_current_layer.NextLayer.relu_parameter = t_relu_coefficient;
      t_current_layer = t_current_layer.NextLayer;
    }
    t_input_layer.AddLayer(t_output_layer);

    const string imagesFileName = "train-images-idx3-ubyte";
    const string labelsFileName = "train-labels-idx1-ubyte";

    var imagesFS = new FileStream(imagesFileName, FileMode.Open);
    var labelsFS = new FileStream(labelsFileName, FileMode.Open);

    BinaryReader imagesBR = new BinaryReader(imagesFS);
    BinaryReader labelsBR = new BinaryReader(labelsFS);

    // load each one into a 28x28 matrix and feed it to the NN
    // set up images
    int magicNumber1 = imagesBR.ReadInt32();
    int numberOfImages = imagesBR.ReadInt32();
    int rowNumber = imagesBR.ReadInt32();
    int colNumber = imagesBR.ReadInt32();

    // set up labels
    int magicNumber2 = labelsBR.ReadInt32();
    int numberOfLabels = labelsBR.ReadInt32();

    byte[][] pixels = new byte[28][];
    for(int i = 0; i < pixels.Length; i++) { pixels[i] = new byte[28]; }

    for(int x = 0; x < 60000; x++) {
      for(int y = 0; y < 28; y++) {
        for(int z = 0; z < 28; z++) {
          byte px = imagesBR.ReadByte();
          pixels[y][z] = px;
        }
      }
      byte label = labelsBR.ReadByte();
      DigitImage dImage = new DigitImage(pixels, label);
      dImage.ConvertBinary();
      t_input_layer.LoadImage(dImage);
      t_input_layer.FeedForward();

      if((x+1)% 10 == 0) {
        Console.WriteLine($"Image: {x+1} Loaded");
        Console.WriteLine($"Loss:{t_output_layer.Loss}");
      }
    }

    // close files
    imagesFS.Close(); imagesBR.Close();
    labelsFS.Close(); labelsBR.Close();
  }

  private void draw_MouseDown(object sender, MouseEventArgs e) {
    mouse_down = true;
    last_point = e.Location;
  }

  private void draw_MouseMove(object sender, MouseEventArgs e) {
    if(last_point != Point.Empty) {
      if(mouse_down) {
        using(Graphics g = Graphics.FromImage(canvas.Image)) {
          Size size = new Size(e.Location.X - last_point.X, e.Location.Y - last_point.Y);
          Rectangle rect = new Rectangle(new Point(e.X,e.Y),size);
          g.DrawEllipse(new Pen(Color.Black,8), rect);
        }
        canvas.Refresh();
        last_point = e.Location;
      }
    }
  }

  private void draw_MouseUp(object sender, MouseEventArgs e) {
    mouse_down = false;
    last_point = Point.Empty;
  }

  private void ImportWeights(string path) {
    StreamReader reader = new StreamReader(path);
    Layer current_layer = networkOutputLayer;
    char ch;

    while((current_layer = current_layer.PreviousLayer) != null) {
      for(int i = 0; i < current_layer.Weights.Length; i++) {
        for(int j = 0; j < current_layer.Weights[i].Length; j++) {
          string current_number = "";
          while((ch = (char)reader.Read()) != ',') {
            current_number += ch;
          }
          current_layer.Weights[i][j] = float.Parse(current_number);
        }
      }
    }
    reader.Close();
  }

  public UI() {
    this.canvas = new PictureBox();
    this.Size = new Size(screen.Width, screen.Height);
    this.mouse_down = false;
    this.canvas.Size = new Size(280,280);
    this.canvas.Location = new Point(350,120);
    this.canvas.BorderStyle = BorderStyle.FixedSingle;
    this.canvas.Image = new Bitmap(canvas.Width, canvas.Height);

    InitialiseNeuralNetwork();
    ImportWeights("weights.txt");
    InitialiseTabs();

    this.canvas.MouseDown += new MouseEventHandler(draw_MouseDown);
    this.canvas.MouseMove += new MouseEventHandler(draw_MouseMove);
    this.canvas.MouseUp += new MouseEventHandler(draw_MouseUp);
  }
}
