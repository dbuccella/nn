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
using System.Windows.Media;
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
            _backgroundWorker.RunWorkerAsync();
        }

        private void ProgressChanged(object sender, ProgressChangedEventArgs e)
        {
            MainViewModel viewModel = (MainViewModel)DataContext;
            viewModel.Points.Add(new OxyPlot.DataPoint(e.ProgressPercentage, (double) e.UserState));
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
        private void DoWork(object sender, DoWorkEventArgs e)
        {
            mlp net = new mlp(2, 3, 2, 1);
            math.Matrix x = new math.Matrix(DataSets.circ_X);
            math.Matrix y = new math.Matrix(DataSets.circ_Y);
            //y.Map((v) => (v == 0.0) ? -1.0 : 1.0);

            net.Train(x, y, _backgroundWorker.ReportProgress);
            /*
            math.Matrix vx = new math.Matrix(DataSets.circ_VX);
            math.Matrix vy = new math.Matrix(DataSets.circ_VY);
            //vy.Map((v) => (v == 0.0) ? -1.0 : 1.0);

            net.Verify(x, y);
            net.Verify(vx, vy);
            */
        }
    }
}
