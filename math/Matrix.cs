using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace math
{
    [Serializable]
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

        public static Matrix Load(string filename)
        {
            Matrix r = null;
            using (Stream stream = File.Open(filename, FileMode.Open))
            {
                var bformatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                r = (Matrix) bformatter.Deserialize(stream);
            }
            return r;
        }

        public static void Save(Matrix r, string filename)
        {
            using (Stream stream = File.Open(filename, FileMode.Create))
            {
                var bformatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                bformatter.Serialize(stream, r);
            }
        }

        private Matrix()
        {
            
            _rows = 0;
            _columns = 0;
            _data = null;
        }

        public Matrix(int rows, int columns)
        {
            _rows = rows;
            _columns = columns;
            _data = new double[rows * columns];
        }
        public Matrix(int columns)
        {
            _rows = 1;
            _columns = columns;
            _data = new double[columns];
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

        public Matrix Clone()
        {
            Matrix r = new Matrix(this);
            return r;
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

        public Matrix RowSlice(int rowStart, int rowCount)
        {
            if ((rowStart + rowCount) > _rows)
                throw new Exception("Slice too large");
            if (rowStart >= _rows)
                throw new Exception("Invalid slice specification");
            Matrix r = new Matrix(rowCount, _columns);
            Array.Copy(_data, rowStart * _columns, r._data, 0, rowCount*_columns);
            return r;
        }

        public void AppendRows(Matrix src, int rowStart, int rowCount)
        {
            if ((rowStart + rowCount) > src._rows)
                throw new Exception("Slice too large");
            if (rowStart >= src._rows)
                throw new Exception("Invalid slice specification");
            if (_columns != src._columns)
                throw new Exception("Not the same column size");
            Array.Resize<double>(ref _data, (_rows + rowCount) * _columns);
            Array.Copy(src._data, rowStart * _columns, _data, _rows*_columns, rowCount * _columns);
            _rows += rowCount;
        }

        public void CopyRows(Matrix src, int srcRowStart, int rowCount, int destRowStart)
        {
            if ((srcRowStart + rowCount) > src._rows)
                throw new Exception("Slice too large");
            if (srcRowStart >= src._rows)
                throw new Exception("Invalid slice specification");
            if (_columns != src._columns)
                throw new Exception("Not the same column size");


            Array.Resize<double>(ref _data, (_rows + rowCount) * _columns);
            Array.Copy(src._data, rowStart * _columns, _data, _rows * _columns, rowCount * _columns);
            _rows += rowCount;
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
                _data[i] *= a._data[i];
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
                _data[i] *= a;
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
            return r.Divide(b);
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

        public void Print(string label="", TextWriter stream = null)
        {
            TextWriter s = stream ?? Console.Out;
            if (label != "")
                s.WriteLine(label);
            s.WriteLine("[{0},{1}]", _rows, _columns);
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _columns; j++)
                {
                    s.Write("{0:N6}   ", this[i, j]);
                }
                s.WriteLine();
            }
            s.WriteLine();
        }

        public void PrintFn(TextWriter f, string msg)
        {
            Print(msg, f);
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

        public double Error()
        {
            return Math.Sqrt(SquaredError());
        }
        public double Mean()
        {
            return _data.Average();
        }

        public double Stdv()
        {
            double mean = _data.Average();
            double sum = 0.0;
            for (int i = 0; i < _data.Length; i++)
                sum += Math.Pow(_data[i] - mean, 2.0);
            return Math.Sqrt(sum / _data.Length);
        }

        public Matrix ColumnSum()
        {
            Matrix r = new Matrix(_columns);
            r.FillZero();
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _columns; j++)
                {
                    r[0, j] += this[i, j];
                }
            }
            return r;
        }

        public Matrix ColumnAvg()
        {
            Matrix r = ColumnSum();
            return r.Map((x) => { return x / _rows; });
        }

        public Matrix ColumnStdv()
        {
            Matrix r = ColumnAvg();
            Matrix diff = Clone().RowOp(r, (u, v) => { return ((u - v) * (u - v)); });
            return diff.ColumnSum().Map((u) => { return Math.Sqrt(u / diff.Rows); });
        }

        public Matrix ColumnStdv(Matrix meanRow)
        {
            Matrix diff = Clone().RowOp(meanRow, (u, v) => { return ((u - v) * (u - v)); });
            return diff.ColumnSum().Map((u) => { return Math.Sqrt(u / diff.Rows); });
        }

        public Matrix ColumnNormalize(Matrix meanRow, Matrix stdvCol)
        {
            RowOp(meanRow, (x, mean) => { return x - mean; });
            RowOp(stdvCol, (x, stdv) => { return x/stdv; });
            return this;
        }

        public Matrix RowSum()
        {
            Matrix r = new Matrix(_rows, 1);
            r.FillZero();
            for (int i = 0; i < _rows; i++)
            {
                for (int j = 0; j < _columns; j++)
                {
                    r[i, 0] += this[i, j];
                }
            }
            return r;
        }

        public delegate double MapFn2(double x, double y);

        public Matrix RowOp(Matrix r, MapFn2 fn)
        {
            if (this._columns != r._columns)
                throw new Exception(String.Format("Incompatible dimensions left({0},{1}) , right({2},{3})", 
                    this._rows, this._columns, r._rows, r._columns));
            for (int i = 0; i < _rows; i++)
            {
                int k = i * _columns;
                for (int j = 0; j < _columns; j++)
                {
                    _data[k + j] = fn(_data[k + j], r._data[j]);
                }
            }
            return this;
        }

        public Matrix ColumnOp(Matrix c, MapFn2 fn)
        {
            if (this._rows != c._rows)
                throw new Exception(String.Format("Incompatible dimensions left({0},{1}) , right({2},{3})",
                    this._rows, this._columns, c._rows, c._columns));
            for (int j = 0; j < _columns; j++)
            {
                for (int i = 0; i < _rows; i++)
                {
                    this[i, j] = fn(this[i, j], c[i, 0]);
                }
            } 
            return this;
        }

    }
}
