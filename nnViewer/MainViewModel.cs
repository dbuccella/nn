using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using OxyPlot;

namespace nnViewer
{
    public class MainViewModel
    {
        public MainViewModel()
        {
            this.Title = "Loss Function";
            this.Points = new List<DataPoint>();
            this.DPoints = new List<DataPoint>();
        }

        public string Title { get; private set; }

        public IList<DataPoint> Points { get; private set; }
        public IList<DataPoint> DPoints { get; private set; }

    }


}
