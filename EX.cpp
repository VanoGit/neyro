#include <math.h>
#include <vector>
#include <fstream>
#include <string>
#include <algorithm>
#include <iomanip>
#include <iostream>

using namespace std;

long double fRand(long double fMin, long double fMax)
{

  long double f = (long double)rand() / RAND_MAX;
  return fMin + f * (fMax - fMin);
}

long double Active(long double arg) {
  long double rez = 1 / (1 + exp(-arg));
  return rez;
}


long double ActiveD(long double arg) {
  long double rez = Active(arg)*(1 - Active(arg));
  return rez;
}

class neyron {
private:
  vector<long double> weights;
  long double out = 0;
public:
  long double use(vector<long double>& lefters) {
    out = 0;
    for (int i = 0; i<lefters.size(); ++i) {
      out += lefters[i] * weights[i];
    }

    return out;
  }

  void ChangeWeight(int num, long double minus) {
    weights[num] -= minus;
  }

  void AddWeight(long double newweight) {
    weights.push_back(newweight);
  }

  long GetWeightsSize() {
    return weights.size();
  }

  long double GetOut() {
    return out;
  }

  long double GetWeight(int num) {
    return weights[num];
  }
};

class layer1 {
private:
  vector<neyron> neyrons;
  vector<long double> outs;
  vector<long double> sigm_shtrix;
  vector<long double> margins;

public:
  void start(vector<long double>& ins) {
    for (int i = 0; i<neyrons.size(); ++i) {
      long double out_it = neyrons[i].use(ins);
      outs.push_back(Active(out_it));
      sigm_shtrix.push_back(ActiveD(out_it));
    }

  }

  void inic(long double neyro_count, long double matrix_width) {
    for (int i = 0; i<neyro_count; ++i) neyrons.push_back(neyron());
    for (int i = 0; i<neyrons.size(); ++i) {
      for (int ii = 0; ii<matrix_width; ++ii) neyrons[i].AddWeight(fRand(-0.5, 0.5));
    }
  }

  void CorrectWeights(vector<long double>& in, long double LearnTemp) {
    for (int i = 0; i < neyrons.size(); ++i) {
      for (int k = 0; k < neyrons[i].GetWeightsSize(); ++k) {
        neyrons[i].ChangeWeight(k, LearnTemp*margins[i] * sigm_shtrix[i] * in[k]);
      }
    }
  }

  vector<long double>& GetOuts() {
    return outs;
  }

  void Cleaner() {
    outs.clear();
    sigm_shtrix.clear();
    margins.clear();
  }

  int GetNeyroCount() {
    return neyrons.size();
  }

  void PushMargin(long double mar) {
    margins.push_back(mar);
  }
};

class layer2 {
private:
  vector<neyron> neyrons;
  vector<long double> outs;
  vector<long double> sigm_shtrix;
  vector<long double> margins;
  vector<long double> rrr;
  long double sum = 0;

public:
  void start(vector<long double>& ins, vector<long double>& answers) {
    for (int i = 0; i<neyrons.size(); ++i) {
      long double out_it = neyrons[i].use(ins);
      sum += exp(out_it);
    }
    for (int i = 0; i<neyrons.size(); ++i) {
      outs.push_back(exp(neyrons[i].GetOut()) / sum);
      sigm_shtrix.push_back((exp(neyrons[i].GetOut()) / sum)*(1 - exp(neyrons[i].GetOut()) / sum));
      margins.push_back(exp(neyrons[i].GetOut()) / sum - answers[i]);
    }
    sum = 0;
  }

  void inic(long double neyro_count, long double hid_count) {
    for (int i = 0; i<neyro_count; ++i) neyrons.push_back(neyron());
    for (int i = 0; i<neyrons.size(); ++i) {
      for (int ii = 0; ii<hid_count; ++ii) neyrons[i].AddWeight(fRand(-0.5, 0.5));
    }
  }

  void CorrectWeights(vector<long double> prev_outs, long double LearnTemp) {
    for (int i = 0; i < neyrons.size(); ++i) {
      for (int k = 0; k < neyrons[i].GetWeightsSize(); ++k) {
        neyrons[i].ChangeWeight(k, LearnTemp*margins[i] * sigm_shtrix[i] * prev_outs[k]);
      }
    }
  }
  void Cleaner() {
    outs.clear();
    sigm_shtrix.clear();
    margins.clear();
  }

  int GetNeyroCount() {
    return neyrons.size();
  }

  long double GetNeyroWeight(int neyroId, int weightId) {
    return neyrons[neyroId].GetWeight(weightId);
  }

  long double GetMargin(int num) {
    return margins[num];
  }

  long double GetSigm(int num) {
    return sigm_shtrix[num];
  }

  long double GetOut(int num) {
    return outs[num];
  }

  int GetOutSize() {
    return outs.size();
  }
};



class NeyroNet {
private:
  layer1 lay1;
  layer2 lay2;
  int neyron_count;
  long double LearnTemp = 0.1;

public:
  void Inic(long double hidcount, long double lastlay, long double matrix_width) {
    neyron_count = hidcount;
    lay1.inic(hidcount, matrix_width);
    lay2.inic(lastlay, hidcount);
  }

  void WeightCorrect(vector<long double>& in) {
    //cout << *min_element(lay2.margins.begin(), lay2.margins.end()) << " " << *max_element(lay2.margins.begin(), lay2.margins.end()) << "\n";
    cout << "Corrected";
    lay1.CorrectWeights(in, LearnTemp);
    lay2.CorrectWeights(lay1.GetOuts(), LearnTemp);
  }

  void Train(vector<long double>& train_set, vector<long double>& answ) {
    lay1.start(train_set);
    lay2.start(lay1.GetOuts(), answ);
    StepBack();
    WeightCorrect(train_set);
    cout << "\n";
    lay1.Cleaner();
    lay2.Cleaner();

  }

  void StepBack() {
    for (int i = 0; i < lay1.GetNeyroCount(); ++i) {
      long double sigma_sh = 0;
      for (int k = 0; k < lay2.GetNeyroCount(); ++k) {
        sigma_sh += lay2.GetMargin(k) * lay2.GetSigm(k) * lay2.GetNeyroWeight(k, i);
      }
      lay1.PushMargin(sigma_sh);
    }
  }

  int TryIt(vector<long double>& train_set, vector<long double>& answ) {
    lay1.start(train_set);
    lay2.start(lay1.GetOuts(), answ);
    int predict = -1;
    long double predictVal = -10;
    for (int i = 0; i < lay2.GetOutSize(); ++i) {
      cout << i << ": " << lay2.GetOut(i) << "\n";
      if (lay2.GetOut(i) > predictVal) {
        predictVal = lay2.GetOut(i);
        predict = i;
      }
    }
    lay1.Cleaner();
    lay2.Cleaner();
    return predict;
  }

};

int main()
{

  ifstream f("mnist_train.csv");
  vector<long double> pixels;
  vector<long double> answ;
  int counter = 2000;
  string zn;
  getline(f, zn);
  NeyroNet n;
  n.Inic(150, 10, 784);

  while (counter>0)
  {
    string num;
    getline(f, num, ';');
    for (int i = 0; i < 783; ++i) {
      getline(f, zn, ';');
      pixels.push_back(stoi(zn));
    }

    getline(f, zn);
    pixels.push_back(stoi(zn));
    counter--;
    for (int ii = 0; ii < pixels.size(); ++ii) {
      pixels[ii] = Active(pixels[ii]);
    }

    answ = { 0,0,0,0,0,0,0,0,0,0 };
    answ[stoi(num)] = 1;
    cout << 2000 - counter << " ";
    n.Train(pixels, answ);
    pixels.clear();
  }

  cout << "-----------------------------------\n";
  counter = 1000;
  int good = 0;
  int bad = 0;
  while (counter > 0) {
    string num;
    getline(f, num, ';');
    for (int i = 0; i < 783; ++i) {
      getline(f, zn, ';');
      pixels.push_back(stoi(zn));
    }

    getline(f, zn);
    pixels.push_back(stoi(zn));
    counter--;
    cout << "NUM IS " << num << "\n";
    for (int ii = 0; ii < pixels.size(); ++ii) {
      pixels[ii] = Active(pixels[ii]);
    }
    if (stoi(num) == n.TryIt(pixels, answ)) good += 1;
    cout << "------------------------" << "\n";

    pixels.clear();
  }
  cout << "good: " << good << "\n" << "accuracy: " << good / 10 << "%";



  return 0;
}
