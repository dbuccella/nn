using math;
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace nnViewer
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private Random _gen;
        private BackgroundWorker _backgroundWorker = new BackgroundWorker();
        double _low = 0;
        double _high = 0;
        double _alpha = 0;
        bool _cancel = false;

        public MainWindow()
        {
            _gen = new Random(DateTime.Now.Millisecond);
            InitializeComponent();
            _backgroundWorker.WorkerReportsProgress = true;
            _backgroundWorker.ProgressChanged += ProgressChanged;
            _backgroundWorker.DoWork += DoWork;
            _backgroundWorker.RunWorkerCompleted += BackgroundWorker_RunWorkerCompleted;
        }

        private void button_Click(object sender, RoutedEventArgs e)
        {
            _low = Double.Parse(fromTxt.Text);
            _high = Double.Parse(toTxt.Text);
            _alpha = Double.Parse(alphaTxt.Text);
            _cancel = false;


            MainViewModel viewModel = (MainViewModel)DataContext;
            viewModel.Points.Clear();
            viewModel.DPoints.Clear();
            MyPlotView.InvalidatePlot(true);
            _backgroundWorker.RunWorkerAsync();
        }

        private void ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            MainViewModel viewModel = (MainViewModel)DataContext;
            PlotData p = (PlotData)e.UserState;
            viewModel.Points.Add(new OxyPlot.DataPoint(p.LossX, p.LossY));
            //viewModel.DPoints.Add(new OxyPlot.DataPoint(p.DeltaX, p.DeltaY));
            MyPlotView.InvalidatePlot(true);
        }

        private void BackgroundWorker_RunWorkerCompleted(object sender, RunWorkerCompletedEventArgs e)
        {
            /*
                           System.Windows.Forms.MessageBox.Show(
                    String.Format("Successful = {0} | Elapsed time= {1} ms | Req Avg = {2:N0} ms",
                    _runner.TransactionCount, _runner.ElapsedTime, 
                    1.0 * _runner.ElapsedTime / ((_runner.TransactionCount == 0) ? 1 : _runner.TransactionCount)),
                    "Success",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Information); 
            */
            MyPlotView.InvalidatePlot(true);
        }
        private void DoWork_0(object sender, DoWorkEventArgs e)
        {
            ///*
            mlp net = new mlp(2, 2, 2, 1, 0.001);
            net.InitLow = _low;
            net.InitHigh = _high;
           
            //math.Matrix x = new math.Matrix(DataSets.circ_X);
            //math.Matrix y = new math.Matrix(DataSets.circ_Y);
            //y.Map((v) => (v == 0.0) ? -1.0 : v);

            math.Matrix x = new math.Matrix(DataSets.polyX);
            math.Matrix y = new math.Matrix(DataSets.polyY);

            Matrix mean = x.ColumnAvg();
            Matrix stdv = x.ColumnStdv();
            x.ColumnNormalize(mean, stdv); 

            TrainResult r = net.Train2(x, y, 50000, _alpha, 10, ref _cancel, _backgroundWorker.ReportProgress);
            /*
            math.Matrix vx = new math.Matrix(DataSets.circ_VX);
            math.Matrix vy = new math.Matrix(DataSets.circ_VY);
            vy.Map((v) => (v == 0.0) ? -1.0 : v);
            vx.ColumnNormalize(mean, stdv);
            VerifyResult vr = net.Verify(vx, vy, 0.2);
            */
            VerifyResult vr = net.Verify(x, y, 0.2);
            MessageBox.Show(String.Format("Epochs= {0} | Error = {1:N5} |\n v.Error = {2:N5} | Accuracy = {3:N5}", 
                r.Epochs, r.Error, vr.Error, vr.Accuracy), "Training Complete");

        }
        private void DoWork(object sender, DoWorkEventArgs e)
        {
            Matrix x = Matrix.Load(@"c:\data\params.mat");
            Matrix y = Matrix.Load(@"c:\data\labels2.mat");
            mlp net = new mlp(x.Columns, 50, 2, 10, 0.001);
            net.InitLow = _low;
            net.InitHigh = _high;
            TrainResult r = net.Train2(x, y, 1000, _alpha, 10, ref _cancel, _backgroundWorker.ReportProgress);
            MessageBox.Show(String.Format("Epochs= {0} | Error = {1:N5}",
                r.Epochs, r.Error), "Training Complete");
        }

        private void cancelBtn_Click(object sender, RoutedEventArgs e)
        {
            _cancel = true;
        }
    }
}
