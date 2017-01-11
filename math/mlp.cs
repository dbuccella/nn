using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace math
{
    public class mlp
    {
        Matrix[] w;
        Matrix[] a;
        int _inpSz;
        int _hiddenNodes;
        int _hiddenLayers;
        int _outSz;

        public mlp(
                int inpSz,
                int hiddenNodes,
                int hiddenLayers,
                int outSz)
        {
            _inpSz = inpSz;
            _hiddenNodes = hiddenNodes;
            _hiddenLayers = hiddenLayers;
            _outSz = outSz;
            w = new Matrix[_hiddenLayers];
            a = new Matrix[_hiddenLayers];
            // build weights matrices
            w[0] = new Matrix(_hiddenNodes, _inpSz);
            for (int i = 1; i < _hiddenLayers-1; i++)
                w[i] = new Matrix(_hiddenNodes, _hiddenNodes);
            w[_hiddenLayers - 1] = new Matrix(_outSz, _hiddenNodes);
        }

        public void InitWeights()
        {
            for (int i = 0; i < _hiddenNodes; i++)
                w[i].FillRandom(-0.5, 0.5);
        }
        public double FF(Matrix x)
        {
            a[0] = x * w[0];
            return 0.0;
        }

    }
}
