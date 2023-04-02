using System;
using System.IO;
using System.Collections.Generic;
/*
  PROJECT DETAILS
  128x128 image with a number between 0-9
  Recognise the number in that image
  INPUT: 128x128 matrix
  OUTPUT: 10 possible outcomes
*/

public class Program {
  public static void Main(string[] args) {
    InputLayer networkInputLayer = new InputLayer(784,512);
    Layer hidden1 = new Layer(512,256);
    Layer hidden2 = new Layer(256,10);
    OutputLayer networkOutputLayer = new OutputLayer();

    networkInputLayer.NextLayer = hidden1;
    hidden1.PreviousLayer = networkInputLayer;
    hidden1.NextLayer = hidden2;
    hidden2.PreviousLayer = hidden1;
    hidden2.NextLayer = networkOutputLayer;
    networkOutputLayer.PreviousLayer = hidden2;

    // open up a folder with a lot of 28x28 images of numbers
    // MNIST database of handwritten digits
    const string imagesFileName = "train-images-idx3-ubyte";
    const string labelsFileName = "train-labels-idx1-ubyte";

    var imagesFS = new FileStream(imagesFileName, FileMode.Open);
    var labelsFS = new FileStream(labelsFileName, FileMode.Open);

    BinaryReader imagesBR = new BinaryReader(imagesFS);
    BinaryReader labelsBR = new BinaryReader(labelsFS);

    // load each one into a 28x28 matrix and feed it to the NN
    // set up images
    int numberOfImages = imagesBR.ReadInt32();
    int magicNumber1 = imagesBR.ReadInt32();

    int rowNumber = imagesBR.ReadInt32();
    int colNumber = imagesBR.ReadInt32();

    // set up labels
    int magicNumber2 = labelsBR.ReadInt32();
    int numberOfLabels = labelsBR.ReadInt32();

    byte[][] pixels = new byte[28][];
    for(int i = 0; i < pixels.Length; i++) { pixels[i] = new byte[28]; }

    for(int x = 0; x < numberOfImages; x++) {
      for(int y = 0; y < 28; y++) {
        for(int z = 0; z < 28; z++) {
          byte px = imagesBR.ReadByte();
          pixels[y][z] = px;
        }
      }
      byte label = labelsBR.ReadByte();
      DigitImage dImage = new DigitImage(pixels, label);
      networkInputLayer.LoadImage(dImage);
      networkInputLayer.FeedForward();
      networkOutputLayer.BackPropagate();
      Console.WriteLine($"Image: {x+1} Loaded");
      Console.ReadLine();
    }

    // close files
    imagesFS.Close(); imagesBR.Close();
    labelsFS.Close(); labelsBR.Close();
  }
}
public class DigitImage
{
    public byte[][] pixels;
    public byte label;

    public DigitImage(byte[][] pixels, byte label)
    {
      this.pixels = new byte[28][];
      for(int i = 0; i < this.pixels.Length; ++i) {
        this.pixels[i] = new byte[28];
      }

      for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
          this.pixels[i][j] = pixels[i][j];
        }
      }

      this.label = label;
    }

    public override string ToString()
    {
      string s = "";
      for (int i = 0; i < 28; ++i)
      {
        for (int j = 0; j < 28; ++j)
        {
          if (this.pixels[i][j] == 0) {
            s += " "; // white
          }
          else if (this.pixels[i][j] > 200) {
            s += "O"; // black
          }
          else {
            s += "."; // gray
          }
        }
        s += "\n";
      }
      s += this.label.ToString();
      return s;
    } // ToString
}

public class Layer {
  public float[] Inputs;
  public float[][] Weights;
  protected float Bias;
  public Layer NextLayer;
  public Layer PreviousLayer;

  protected float[] InitialiseWeights(int n) {
    Random rnd = new Random();
    float[] _weights = new float[n];
    for(int i = 0; i < n; i++) {
      _weights[i] = (float)(2*rnd.NextDouble() - 1);
    }
    return _weights;
  }

  protected float ReLU(float x) => (float)(x > 0 ? x : 0);

  protected float DotProduct(float[] _array1, float[] _array2) {
    float sum = 0;
    if(_array1.Length == _array2.Length) {
      for(int i = 0; i < _array1.Length; i++) {
        sum += _array1[i] * _array2[i];
      }
    }
    return sum;
  }

  public float FeedForward(float[] new_inputs, byte _label) {
    if(this.NextLayer == null) {
      float error = ((OutputLayer)this).FeedForward(new_inputs, _label);
      return error;
    }
    float[] results = new float[this.Weights.Length];
    // perform dotproduct operation
    for(int i = 0; i < this.Weights.Length; i++) {
      results[i] = DotProduct(this.Inputs, this.Weights[i]) + Bias;
    }
    // normalise results
    float[] normalisedResults = Statistics.NormaliseDataSet(results);
    // pass normalised results through relu function
    float[] activatedResults = new float[results.Length];
    for(int n = 0; n < results.Length; n++) {
      activatedResults[n] = ReLU(normalisedResults[n]);
    }
    // set next layer inputs to these outputs
    this.NextLayer.Inputs = activatedResults;
    return this.NextLayer.FeedForward(activatedResults, _label);
  }

  public Layer(int weight_count, int next_weight_count) {
    this.Weights = new float[next_weight_count][];
    for(int w = 0; w < this.Weights.Length; w++) {
      // generate an array of random weights for each new neuron
      this.Weights[w] = InitialiseWeights(weight_count);
    }
    this.Bias = 0;
  }
}

public class InputLayer : Layer {
  byte[] PixelNodes;
  byte Label;

  public void LoadImage(DigitImage _image) {
    this.PixelNodes = new byte[28*28];
    int pixelNumber = 0;
    for(int i = 0; i < 28; i++) {
      for(int j = 0; j < 28; j++) {
        this.PixelNodes[pixelNumber] = _image.pixels[i][j];
        pixelNumber++;
      }
    }
    this.Label = _image.label;
  }

  protected float DotProduct(byte[] _nodes, float[] _weights) {
    float sum = 0;
    if(_nodes.Length == _weights.Length) {
      for(int i = 0; i < _nodes.Length; i++) {
        sum += _nodes[i] * _weights[i];
      }
    }
    return sum;
  }

  public float FeedForward() {
    float[] results = new float[this.Weights.Length];
    // perform dotproduct operation
    for(int i = 0; i < this.Weights.Length; i++) {
      results[i] = DotProduct(this.PixelNodes, this.Weights[i]) + Bias;
    }
    // batch normalise results
    float[] normalisedResults = Statistics.NormaliseDataSet(results);
    // pass normalised results through relu function
    float[] activatedResults = new float[results.Length];
    for(int n = 0; n < results.Length; n++) {
      activatedResults[n] = ReLU(normalisedResults[n]);
    }
    // set next layer inputs to these outputs
    this.NextLayer.Inputs = activatedResults;
    return this.NextLayer.FeedForward(activatedResults, this.Label);
  }

  public InputLayer(int weight_count, int next_weight_count) : base(weight_count, next_weight_count) {
    this.Inputs = null;
    this.Weights = new float[next_weight_count][];

    for(int w = 0; w < this.Weights.Length; w++) {
      this.Weights[w] = InitialiseWeights(weight_count);
    }
  }
}

public class OutputLayer : Layer {
  protected float LearningRate;
  protected float[] OutputProbabilities;

  private float[] Softmax(float[] x) {
    // calculate sum of e^(x_i)
    float sum = 0;
    for(int i = 0; i < x.Length; i++) {
      sum += (float)Math.Exp(x[i]);
    }
    float[] probabilities = new float[x.Length];
    for(int z = 0; z < x.Length; z++) {
      probabilities[z] = (float)(Math.Exp(x[z]) / sum);
    }
    return probabilities;
  }


  private float CrossEntropyLoss(float[] _probabilities, float[] _truth_label) {
    float sum = 0;
    for(int i = 0; i < _probabilities.Length; i++) {
      sum += _truth_label[i]*(float)Math.Log(_probabilities[i]);
    }
    return -sum;
  }

  private int[] GetTruthLabel(float[] _results, byte _label) {
    int[] truth_label = new int[_results.Length];
    for(int t = 0; t < truth_label.Length; t++) {
      if(t == _label) {
        truth_label[t] = 1;
      } else {
        truth_label[t] = 0;
      }
    }
    return truth_label;
  }

  public new float FeedForward(float[] _new_inputs, byte _label) {
    this.OutputProbabilities = Softmax(_new_inputs);
    int[] truth_label = GetTruthLabel(_new_inputs, _label);
    CalculateDerivative(this, truth_label, null);
    return BackPropagate();
  }

  public T[][] MatrixTranspose<T>(T[][] matrix) {
    T[][] transposed_matrix = new T[matrix[0].Length][];
    for(int i = 0; i < transposed_matrix.Length; i++) {
      transposed_matrix[i] = new T[matrix.Length];
      for(int j = 0; j < matrix.Length; j++) {
        transposed_matrix[i][j] = matrix[j][i];
      }
    }
    return transposed_matrix;
  }

  public T[][] MatrixMultiplication<T>(T[][] mat1, T[][] mat2) {
    T[][] result = new T[mat1.Length][];
    for(int i = 0; i < mat1.Length; i++) {
      result[i] = new T[mat2[0].Length];
      for(int j = 0; j < mat2[0].Length) {
        result[i][j] = mat1[i][j] * mat2[j][i];
      }
    }
    return result;
  }

  public float CalculateDerivative(Layer currentLayer, int[] _truth_label, float[] delta) {
    // List<float> derivatives = new List<float>();
    float[] derivatives;
    if(currentLayer.NextLayer == null) {
      // float loss = CrossEntropyLoss(output_probabilities, _truth_label, truth_label);
      derivatives = new float[this.OutputProbabilities.Length];
      for(int i = 0; i < this.OutputProbabilities.Length; i++) {
        derivatives[i] = this.OutputProbabilities[i] - _truth_label[i];
      }
      return CalculateDerivative(currentLayer.PreviousLayer, _truth_label, derivatives);
    }
    derivatives = new float[currentLayer.Weights.Length];
    // find hidden layer derivative
    // transpose weights
    float[][] transposedWeights = MatrixTranspose(currentLayer.Weights);
    // matrix multiplication
    
    for(int w = 0; w < currentLayer.Weights.Length; w++) {
      //derivatives[w] = DotProduct(currentLayer.Weights[w], delta);
    }
    return 1;
    // find input layer derivative
  }

  public float BackPropagate() {
    return 1;
  }

  public OutputLayer() : base(0,0) {
    this.Weights = null;
    this.NextLayer = null;
    this.LearningRate = 0.2f; // init to low value (0.2-0.5)
  }
}

class Statistics {
  public static float Mean(float[] data_set) {
    float sum = 0;
    for(int i = 0; i < data_set.Length; i++) { sum += data_set[i]; }
    return sum / data_set.Length;
  }

  public static float StandardDeviation(float[] data_set) {
    float mean = Mean(data_set);
    float sumOfSquares = 0;
    for(int i = 0; i < data_set.Length; i++) {
      sumOfSquares += (data_set[i]-mean)*(data_set[i]-mean);
    }

    return (float)Math.Sqrt(sumOfSquares/data_set.Length);
  }

  public static float[] NormaliseDataSet(float[] data_set) {
    float[] normalisedDataSet = new float[data_set.Length];
    float mean = Mean(data_set);
    float stdev = StandardDeviation(data_set);

    for(int i = 0; i < data_set.Length; i++) {
      normalisedDataSet[i] = (data_set[i] - mean) / stdev;
    }
    return normalisedDataSet;
  }
}
