using math;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace tester
{
    class Program
    {
        const double Mu = -0.1;

        static double[,] S = new double[,]  {
            {0.0, 0.0},
            {0.0, 1.0},
            {1.0, 0.0},
            {1.0, 1.0}
            };

        static double[,] XOR = new double[,]  {
            {-1.0},
            {1.0},
            {1.0},
            {-1.0}
            };

        static double[,] X = new double[,] {
        {0.2, 1.020562258},
        {0.43, 1.218506908},
        {0.32, 0.184617471},
        {0.53, 1.534720884},
        {0.59, 0.822312634},
        {0.678, 0.419922912},
        {0.766, -0.216603169},
        {0.854, -0.583984585},
        {0.942, 1.698239806},
        {1.03, 0.953863041},
        {1.118, 1.875333287},
        {1.206, 1.132548225},
        {1.294, 2.058306355},
        {1.382, 0.293087256},
        {1.47, 0.444208613},
        {1.558, 1.892133761},
        {1.646, 0.99},
        {1.734, 0.93},
        {1.822, 0.988834473},
        {1.91, 0.948740604},
        {1.998, 0.908646735},
        {2.086, 0.868552867},
        {2.174, 0.828458998}};

        static double[,] Y = new double[,] {
        {0},
        {0},
        {1},
        {0},
        {0},
        {1},
        {1},
        {1},
        {0},
        {0},
        {0},
        {0},
        {0},
        {1},
        {1},
        {0},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1},
        {1}};

        static double[,] VX = new double[,]{
        {0.473326958,0.032684313},
        {-0.257836836,0.170549968},
        {-0.851918897,-0.23161463},
        {0.989265156,0.744724906},
        {0.681344659,0.437958342},
        {0.710722887,0.636470473},
        {-0.562265989,-0.129908175},
        {-0.866112786,-0.231810514},
        {0.842265815,1.080677695},
        {0.03972403,0.614655844},
        {-0.939167934,-0.543522961},
        {-0.597955993,0.303809349},
        {-0.215915883,0.32955111},
        {-0.737554712,0.181623063},
        };

        static double[,] VY = new double[,] {
            {0},
            {1},
            {1},
            {1},
            {0},
            {0},
            {1},
            {1},
            {1},
            {1},
            {1},
            {0},
            {0},
            {1} };


        static void MainReal(string[] args)
        {
            mlp net = new mlp(2, 3, 3, 1);
            Matrix x = new Matrix(DataSets.circ_X);
            Matrix y = new Matrix(DataSets.circ_Y);
            //y.Map((v) => (v == 0.0) ? -1.0 : 1.0);
            bool cancel = false;
            net.Train(x, y, 100000, 0.001, ref cancel);
            Matrix vx = new Matrix(DataSets.circ_VX);
            Matrix vy = new Matrix(DataSets.circ_VY);
            //vy.Map((v) => (v == 0.0) ? -1.0 : 1.0);

            //net.Verify(x, y);
            net.Verify(vx, vy);
        }

        static void Main7(string[] args)
        {
            mlp net = new mlp(3, 2, 2, 3, 0.001);
            Matrix x = new Matrix(DataSets.class_X);
            Matrix y = new Matrix(DataSets.class_Y);
            y.Map((v) => (v == 0.0) ? -1.0 : 1.0);

            //net.Train(x, y, 150000, 0.05);
            //net.TrainMB(x, y, 100000, 3);
            net.Verify(x, y);
            //y.Print("Y");
            //net.Predict(x);
            /*
            Matrix vx = new Matrix(DataSets.circ_VX);
            Matrix vy = new Matrix(DataSets.circ_VY);
            //vy.Map((v) => (v == 0.0) ? -1.0 : 1.0);

            net.Verify(x, y);
            net.Verify(vx, vy);
            */
        }
        static void Main0(string[] args)
        {
            DataGen.Go2D("c:\\src\\circ.txt", "circ", (x) => Math.Sqrt(20 - Math.Pow(x, 2.0) / 8.0), 50, 0.5, 0.15);
        }
        public static Matrix StdDevColumn(Matrix x)
        {
            Matrix meanCol = x.ColumnSum().Map((u) => { return u / x.Rows; });
            //meanCol.Print("mean");
            Matrix diff = x.Clone().RowOp(meanCol, (u, v) => { return ((u - v) * (u - v)); });
            //diff.Print("sqr diff");
            return diff.ColumnSum().Map((u) => { return Math.Sqrt(u / diff.Rows); });
        }

        static void Main99(string[] args)
        {
            /*
            Matrix x = new Matrix(new double[,]  {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0},
                {7.0, 8.0, 9.0}
            });
            */
            Matrix x = new Matrix(DataSets.circ_X);
            //x.FillRandom(-2, 2);
            //x.Print("x");
            Matrix stdvCol = StdDevColumn(x);
            stdvCol.Print("stdv");
            x.RowOp(stdvCol, (u, v) => { return u - v; }).Print("norm");
            /*
            Matrix m = new Matrix(new double[,]  {
            {2.0, 1.5, 1.0} });
            m.Print("m");

            x.RowOp(m, (u, v) => { return u * v; }).Print("row op *");
            */
            Matrix y = new Matrix(new double[,]  {
            {9.0, 8.0, 7.0},
            {6.0, 5.0, 4.0},
            {3.0, 2.0, 1.0}
            });
            y.Print("y");
            Matrix z = new Matrix(new double[,]  {
            {3 }, { 4 }, { 5} });
            z.Print("z");
            y.ColumnOp(z, (u, v) => { return u / v; }).Print("col op /");

        }

        static void Main_(string[] args)
        {
            Matrix x = new Matrix(new double[,]  {
            {9.0, 8.0, 7.0},
            {6.0, 5.0, 4.0},
            {3.0, 2.0, 1.0}
            });

            Matrix y = new Matrix(new double[,]  {
            {1.5, 0.5, 2.0} });
            Matrix d = new Matrix(new double[,]  {
            {0.3 }, { 0.24 }, { 1.2} });

            y.Dot(x).Print("dot");
            x.Dot(y.Transpose()).Print("T.dot");
            d.Print("d");
            Matrix u = x.Dot(y.Transpose());
            Matrix z = (u * d);
            z.Print("prod");
            z.MapNew((v) => v * 1.2).Print("map");
            z.Print("prod");
            x.Row(0).Print("x");
            y.Print("y");
            Matrix del = x.Row(0) - y;
            del.Print("del");

        }

        static Matrix MakeBatch(Matrix x, int start, int batchSize, Indexer idx)
        {
            // make batch
            Matrix xb = x.Row(idx[start]);
            for (int k = start+1; k < (start+batchSize); k++)
                xb.AppendRows(x, idx[k], 1);
            return xb;
        }
        static bool Compare(Matrix a, Matrix b)
        {
            bool result = true;
            for (int i = 0; i < a.Rows; i++)
            {
                for (int j = 0; j < a.Columns; j++)
                {
                    double aa = a[i, j];
                    double bb = b[i, j];
                    result = result && (a[i,j] == b[i,j]);
                    if (!result)
                        return false;
                }
            }
            return result;
        }

        static void Main77(string[] args)
        {
            List<Matrix> batches = new List<Matrix>();
            Matrix x = new Matrix(DataSets.circ_X);
            x.Print("X");
            x.ColumnSum().Print("X-sum");

            Indexer idx = new Indexer(x.Rows);
            idx.Shuffle();
            int batchSize = 4;
            for (int i = 0; i < x.Rows/batchSize; i++)
            {
                Matrix xb = MakeBatch(x, i * batchSize, batchSize, idx);
                batches.Add(xb);
            }
            int leftover =  x.Rows % batchSize;
            Matrix lb = MakeBatch(x, x.Rows - leftover, leftover, idx);
            batches.Add(lb);
            Matrix n = new Matrix(batches[0]);
            for (int i = 1; i < batches.Count; i++)
            {
                n.AppendRows(batches[i], 0, batches[i].Rows);
            }
            n.Print("n");
            n.ColumnSum().Print("n-sum");
            //xb.Print(String.Format("b[{0}]", i));
            //bool r = Compare(x, n);
            //
        }

        static public void loadMNIST(string fn, string labelsFn, string paramsFn)
        {
            using (var fs = File.OpenRead(fn))
            using (var reader = new StreamReader(fs))
            {
                List<byte> targetLst = new List<byte>();
                List<List<byte>> dataList = new List<List<byte>>();
                // ignore column name line
                reader.ReadLine();
                while (!reader.EndOfStream)
                {
                    List<byte> dataRow = new List<byte>();
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    targetLst.Add(Byte.Parse(values[0]));
                    for (int i = 1; i < values.Length; i++)
                    {
                        dataRow.Add(Byte.Parse(values[i]));
                    }
                    dataList.Add(dataRow);
                }
                // make matrices
                Matrix labels = new Matrix(targetLst.Count, 10);
                labels.FillZero();
                int k = 0;
                foreach(byte item in targetLst)
                {
                    labels[k++, item] = 1.0;
                }
                //serialize labels
                using (Stream stream = File.Open(labelsFn, FileMode.Create))
                {
                    var bformatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                    bformatter.Serialize(stream, labels);
                }
                /*
                //
                Matrix varMat = new Matrix(dataList.Count, dataList[0].Count);
                k = 0;
                int j = 0;
                foreach(List<byte> row in dataList)
                {
                    foreach (byte item in row)
                    {
                        varMat[k, j++] = item;
                    }
                    k++;
                    j = 0;
                }

                //serialize data
                using (Stream stream = File.Open(paramsFn, FileMode.Create))
                {
                    var bformatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                    bformatter.Serialize(stream, varMat);
                }
                */
            }
        }

        static void Main(string[] args)
        {
            //loadMNIST(@"C:\Users\Donato\Downloads\digits-train.csv", @"c:\data\labels2.mat", "./params.mat");
            Matrix y = Matrix.Load(@"c:\data\labels2.mat");
        }


    }
}
