using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace math
{
    public class Matrix
    {
        int _rows;
        int _columns;
        double[] _data;

        public static void EnsureSameDim(Matrix a, Matrix b)
        {
            if ((a._rows != b._rows) || (a._columns != b._columns))
                throw new Exception("Not the same dimension");
        }

        public Matrix(int rows, int columns)
        {
            _rows = rows;
            _columns = columns;
            _data = new double[rows * columns];
            FillZero();
        }
        public Matrix(int columns)
        {
            _rows = 1;
            _columns = columns;
            _data = new double[columns];
            FillZero();
        }

        public Matrix(Matrix a)
        {
            _rows = a._rows;
            _columns = a._columns;
            _data = new double[_rows * _columns];
            Array.Copy(a._data, this._data, _rows * _columns);
        }

        public Matrix(double [,] data)
        {
            _rows = data.GetLength(0);
            _columns = data.GetLength(1);
            _data = new double[_rows * _columns];
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _columns; j++)
                {
                    this[i, j] = data[i, j];
                }
            }
        }

        public double this[int i, int j]
        {
            get
            {
                return _data[i * _columns + j];
            }
            set
            {
                _data[i * _columns + j] = value;
            }
        }
        public int Rows { get { return _rows; } }
        public int Columns { get { return _columns; } }
        public Matrix Row(int i)
        {
            Matrix r = new Matrix(1, _columns);
            Array.Copy(_data, i * _columns, r._data, 0, _columns);
            return r;
        }
        public Matrix Column(int i)
        {
            Matrix r = new Matrix(_rows, 1);
            for (int k = 0; k < _rows; k++)
            {
                r[k, 0] = this[k,i]; 
            }
            return r;
        }

        public Matrix Copy(Matrix a)
        {
            EnsureSameDim(this, a);
            Array.Copy(a._data, this._data, _rows * _columns);
            return this;
        }

        public Matrix Fill(double x)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = x;
            }
            return this;
        }
        public Matrix FillZero()
        {
            Fill(0.0);
            return this;
        }
        public Matrix FillRandom(double min, double max)
        {
            Random gen = new Random(DateTime.Now.Millisecond);
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = (max - min) * gen.NextDouble() + min;
            }
            return this;

        }
        public Matrix Sum(Matrix a)
        {
            EnsureSameDim(this, a);
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] += a._data[i];
            }
            return this;
        }
        public Matrix Substract(Matrix a)
        {
            EnsureSameDim(this, a);
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] -= a._data[i];
            }
            return this;
        }
        public Matrix Multiply(Matrix a)
        {
            EnsureSameDim(this, a);
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] += a._data[i];
            }
            return this;
        }
        public Matrix Divide(Matrix a)
        {
            EnsureSameDim(this, a);
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] /= a._data[i];
            }
            return this;
        }
        public Matrix Sum(double a)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] += a;
            }
            return this;
        }
        public Matrix Substract(double a)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] -= a;
            }
            return this;
        }
        public Matrix Multiply(double a)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] += a;
            }
            return this;
        }
        public Matrix Divide(double a)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] /= a;
            }
            return this;
        }

        public static Matrix operator +(Matrix a, Matrix b)
        {
            EnsureSameDim(a, b);
            Matrix r = new Matrix(a);
            return r.Sum(b);
        }
        public static Matrix operator -(Matrix a, Matrix b)
        {
            EnsureSameDim(a, b);
            Matrix r = new Matrix(a);
            return r.Substract(b);
        }
        public static Matrix operator *(Matrix a, Matrix b)
        {
            EnsureSameDim(a, b);
            Matrix r = new Matrix(a);
            return r.Multiply(b);
        }
        public static Matrix operator /(Matrix a, Matrix b)
        {
            EnsureSameDim(a, b);
            Matrix r = new Matrix(a);
            return r.Multiply(b);
        }

        public Matrix Dot(Matrix a)
        {
            if (this._columns != a._rows)
                throw new Exception(String.Format("Incompatible dimensions left({0},{1}) , right({2},{3})", this._rows, this._columns, a._rows, a._columns));
            Matrix r = new Matrix(_rows, a._columns);
            // dot products of the inner dimensions
            for (int l = 0; l < _rows; l++)
            {
                for (int k = 0; k < a._columns; k++)
                {
                    r[l, k] = 0.0;
                    for (int i = 0; i < _columns; i++)
                    {
                        // we know _columns == a._row
                        r[l, k] += (this[l, i] * a[i, k]);
                    }
                }
            }
            return r;
        }
        public Matrix Transpose()
        {
            Matrix r = new Matrix(this._columns, this._rows);
            for (int i = 0; i < r._rows; i++)
            {
                for (int j = 0; j < r._columns; j++)
                {
                    r[i,j] = this[j,i];
                }
            }
            return r;
        }

        public static Matrix Transpose(Matrix a)
        {
            Matrix r = new Matrix(a._columns, a._rows);
            for (int i = 0; i < r._rows; i++)
            {
                for (int j = 0; j < r._columns; j++)
                {
                    r[i, j] = a[j, i];
                }
            }
            return r;
        }

        public delegate double MapFn1(double x);
        public delegate double MapRCFn1(int row, int col, double x);

        public Matrix MapNew(MapFn1 fn)
        {
            Matrix r = new Matrix(this._rows, this._columns);
            for (int i = 0; i < _data.Length; i++)
            {
                r._data[i] = fn(_data[i]);
            }
            return r;
        }
        public Matrix MapRCNew(MapRCFn1 fn)
        {
            Matrix r = new Matrix(this._rows, this._columns);
            for (int i = 0; i < _data.Length; i++)
            {
                r._data[i] = fn(i / this._columns, i % this._columns, _data[i]);
            }
            return r;
        }
        public Matrix Map(MapFn1 fn)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = fn(_data[i]);
            }
            return this;
        }
        public Matrix MapRC(MapRCFn1 fn)
        {
            for (int i = 0; i < _data.Length; i++)
            {
                _data[i] = fn(i / this._columns, i % this._columns, _data[i]);
            }
            return this;
        }
        public void Print(string label="")
        {
            if (label != "")
                Console.WriteLine(label);
            Console.WriteLine("[{0},{1}]", _rows, _columns);
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _columns; j++)
                {
                    Console.Write("{0:N6}   ", this[i, j]);
                }
                Console.WriteLine();
            }
            Console.WriteLine();
        }

        public double SquaredError()
        {
            double error = 0.0;
            for (int i = 0; i < _data.Length; i++)
            {
                error += (_data[i]*_data[i]);
            }
            return error;
        }
    }
}
