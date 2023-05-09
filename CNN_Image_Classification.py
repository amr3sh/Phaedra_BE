{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "name": "CNN Image Classifier.ipynb",
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "metadata": {
        "id": "CoRxDEF3ycpB"
      },
      "source": [
        "import os\n",
        "from os import listdir, makedirs\n",
        "from os.path import join, exists, expanduser\n",
        " \n",
        "from keras import applications\n",
        "from keras.preprocessing.image import ImageDataGenerator\n",
        "from keras import optimizers\n",
        "from keras.models import Sequential, Model\n",
        "from keras.layers import Dense, GlobalAveragePooling2D\n",
        "from keras.layers import Activation, Dropout, Flatten, Dense\n",
        "from keras import backend as K\n",
        "import tensorflow as tf\n",
        "import cv2\n",
        "import numpy as np\n",
        "from skimage import io\n",
        "import matplotlib.pyplot as plt"
      ],
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "G1EjvZg6z-mv"
      },
      "source": [
        "camera_settings = '/content/drive/MyDrive/Dataset for Camera Settings/'\n",
        "batch_size = 100\n",
        "cm_classes = ['c - ISO Low', '1 - Correct', 'b - Shutter Speed Low', 'd - ISO High', 'a - Shutter Speed High']"
      ],
      "execution_count": 3,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "oUSqZf9O1DWP"
      },
      "source": [
        "# Read Accident Detection Data\n",
        "cm_iso_low=[]\n",
        "cm_correct=[]\n",
        "cm_spd_low=[]\n",
        "cm_iso_high=[]\n",
        "cm_spd_high=[]\n",
        "\n",
        "import glob, os\n",
        "for indv in cm_classes:\n",
        "  os.chdir(camera_settings+\"/\"+indv)\n",
        "  for file in glob.glob(\"*.jpg\"):\n",
        "    if indv == 'c - ISO Low':\n",
        "      image = cv2.imread(camera_settings+\"/\"+indv+\"/\"+file)\n",
        "      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)\n",
        "      cm_iso_low.append(image)\n",
        "    if indv == '1 - Correct':\n",
        "      image = cv2.imread(camera_settings+\"/\"+indv+\"/\"+file)\n",
        "      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)\n",
        "      cm_correct.append(image)\n",
        "    if indv == 'b - Shutter Speed Low':\n",
        "      image = cv2.imread(camera_settings+\"/\"+indv+\"/\"+file)\n",
        "      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)\n",
        "      cm_spd_low.append(image)\n",
        "    if indv == 'd - ISO High':\n",
        "      image = cv2.imread(camera_settings+\"/\"+indv+\"/\"+file)\n",
        "      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)\n",
        "      cm_iso_high.append(image)\n",
        "    if indv == 'a - Shutter Speed High':\n",
        "      image = cv2.imread(camera_settings+\"/\"+indv+\"/\"+file)\n",
        "      image = cv2.resize(image, (128,128), interpolation = cv2.INTER_AREA)\n",
        "      cm_spd_high.append(image)"
      ],
      "execution_count": 24,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "YEl6uVU82IVr"
      },
      "source": [
        "# Generate Labels for each Class in the Data\n",
        "camera_settings_lbls=[]\n",
        "\n",
        "for itm in cm_iso_low:\n",
        "  if itm is not None:\n",
        "    camera_settings_lbls.append([1,0,0,0,0])\n",
        "  else:\n",
        "    print(\"None\")\n",
        "\n",
        "for itm in cm_correct:\n",
        "  if itm is not None:\n",
        "    camera_settings_lbls.append([0,1,0,0,0])\n",
        "  else:\n",
        "    print(\"None\")\n",
        "\n",
        "for itm in cm_spd_low:\n",
        "  if itm is not None:\n",
        "    camera_settings_lbls.append([0,0,1,0,0])\n",
        "  else:\n",
        "    print(\"None\")\n",
        "\n",
        "for itm in cm_iso_high:\n",
        "  if itm is not None:\n",
        "    camera_settings_lbls.append([0,0,0,1,0])\n",
        "  else:\n",
        "    print(\"None\")\n",
        "\n",
        "for itm in cm_spd_high:\n",
        "  if itm is not None:\n",
        "    camera_settings_lbls.append([0,0,0,0,1])\n",
        "  else:\n",
        "    print(\"None\")"
      ],
      "execution_count": 25,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "Ef9KoE8h3CAY"
      },
      "source": [
        "# Merge All The Data\n",
        "# Merge Data\n",
        "camera_settings_imgs=[]\n",
        "\n",
        "for itm in cm_iso_low:\n",
        "  if itm is not None:\n",
        "    camera_settings_imgs.append(itm)\n",
        "  else:\n",
        "    print(\"None\")\n",
        "\n",
        "for itm in cm_correct:\n",
        "  if itm is not None:\n",
        "    camera_settings_imgs.append(itm)\n",
        "  else:\n",
        "    print(\"None\")\n",
        "\n",
        "for itm in cm_spd_low:\n",
        "  if itm is not None:\n",
        "    camera_settings_imgs.append(itm)\n",
        "  else:\n",
        "    print(\"None\")\n",
        "\n",
        "for itm in cm_iso_high:\n",
        "  if itm is not None:\n",
        "    camera_settings_imgs.append(itm)\n",
        "  else:\n",
        "    print(\"None\")\n",
        "\n",
        "for itm in cm_spd_high:\n",
        "  if itm is not None:\n",
        "    camera_settings_imgs.append(itm)\n",
        "  else:\n",
        "    print(\"None\")"
      ],
      "execution_count": 26,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "GlLK09XC3STS"
      },
      "source": [
        "camera_settings_imgs = np.asarray(camera_settings_imgs).astype('float32')\n",
        "camera_settings_lbls = np.asarray(camera_settings_lbls).astype('float32')"
      ],
      "execution_count": 27,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "0tAn26FE3yws",
        "outputId": "e9dd4f02-a57f-4bcf-cab2-a83f2bf05b3a"
      },
      "source": [
        "print(len(camera_settings_imgs))\n",
        "print(len(camera_settings_lbls))"
      ],
      "execution_count": 28,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "500\n",
            "500\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "AoYPc8bg339z"
      },
      "source": [
        "# Split Data into Train and Test\n",
        "from sklearn.model_selection import train_test_split\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(camera_settings_imgs, camera_settings_lbls, test_size=0.2, random_state=42)"
      ],
      "execution_count": 29,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "6xaWYdXZ4EwG"
      },
      "source": [
        "# Model Building"
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "oY4PfND14VYR"
      },
      "source": [
        "## Base Model"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "QDUE2m-q4C2b"
      },
      "source": [
        "# Base Model\n",
        "# base model\n",
        "from keras.models import Sequential\n",
        "from keras.layers import Dense, Conv2D, Flatten\n",
        "#create model\n",
        "model_1 = Sequential()\n",
        "#add model layers\n",
        "model_1.add(Conv2D(64, kernel_size=2, activation='relu', input_shape=(128,128,3)))\n",
        "model_1.add(Conv2D(32, kernel_size=2, activation='relu'))\n",
        "model_1.add(Flatten())\n",
        "model_1.add(Dense(5, activation='softmax'))"
      ],
      "execution_count": 40,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "mCSXCkmi4JA6"
      },
      "source": [
        "#compile base model using accuracy to measure model performance\n",
        "model_1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])"
      ],
      "execution_count": 41,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "9LIea1WS4OEG",
        "outputId": "08a97984-2393-4e60-fd19-dede124abd93"
      },
      "source": [
        "# Train Base Model\n",
        "history_base = model_1.fit(X_train, y_train, epochs=10, shuffle = True, verbose = 1, validation_split=0.2)"
      ],
      "execution_count": 42,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/10\n",
            "10/10 [==============================] - 11s 1s/step - loss: 5250.9921 - accuracy: 0.2198 - val_loss: 998.5729 - val_accuracy: 0.1875\n",
            "Epoch 2/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 561.6751 - accuracy: 0.2568 - val_loss: 44.5295 - val_accuracy: 0.2250\n",
            "Epoch 3/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 6.7577 - accuracy: 0.4570 - val_loss: 5.3810 - val_accuracy: 0.2500\n",
            "Epoch 4/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 0.6105 - accuracy: 0.7754 - val_loss: 3.6714 - val_accuracy: 0.2375\n",
            "Epoch 5/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 0.4851 - accuracy: 0.8939 - val_loss: 4.1660 - val_accuracy: 0.2250\n",
            "Epoch 6/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 0.3833 - accuracy: 0.9233 - val_loss: 4.5703 - val_accuracy: 0.2125\n",
            "Epoch 7/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 0.2629 - accuracy: 0.9619 - val_loss: 5.1606 - val_accuracy: 0.1875\n",
            "Epoch 8/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 0.2350 - accuracy: 0.9569 - val_loss: 6.2100 - val_accuracy: 0.1875\n",
            "Epoch 9/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 0.2014 - accuracy: 0.9497 - val_loss: 7.2974 - val_accuracy: 0.2000\n",
            "Epoch 10/10\n",
            "10/10 [==============================] - 10s 1s/step - loss: 0.1166 - accuracy: 0.9736 - val_loss: 8.0934 - val_accuracy: 0.2125\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "W64nRbcJ8hEa",
        "outputId": "4f0f8abe-f684-4512-bae3-5ee19c41ae47"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "# summarize history for accuracy\n",
        "plt.plot(history_base.history['accuracy'], label='Training Accuracy')\n",
        "plt.plot(history_base.history['val_accuracy'], label='Validation Accuracy')\n",
        "plt.title('Accuracy')\n",
        "plt.ylabel('Accuracy(Training and Validation)')\n",
        "plt.xlabel('epoch')\n",
        "plt.legend(loc='lower right')\n",
        "#plt.legend(['Training'],['Validation'], loc='upper left')\n",
        "plt.title('Training and Validation Accuracy')\n",
        "\n",
        "#plt.legend(['Pretrained'], loc='upper left')\n",
        "plt.show()\n",
        "# summarize history for loss\n",
        "plt.plot(history_base.history['loss'], label='Training Loss')\n",
        "plt.plot(history_base.history['val_loss'], label='Validation Loss')\n",
        "plt.title('Loss')\n",
        "plt.ylabel('Loss(Training and Validation)')\n",
        "plt.xlabel('epoch')\n",
        "#plt.legend(loc='upper right')\n",
        "plt.title('Training and Validation Loss')\n",
        "#plt.legend(loc='upper right')\n",
        "plt.legend(loc='upper right')\n",
        "plt.show()\n"
      ],
      "execution_count": 45,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdd3xV9f348dc7i5DBCAl7BJDpACRsF4KKFaFuaVFwYa1W0Wpr/apVq7+2Sq1aFUVxoFTcFpWCAuIChTCqbEJYYYYEMkhC1vv3xzkJl5Dc3ITc3Iz38/G4D+6Z930vcN7nfD6f8z6iqhhjjGm8ggIdgDHGmMCyRGCMMY2cJQJjjGnkLBEYY0wjZ4nAGGMaOUsExhjTyFkiMMcRkf+KyKSaXjeQRGS7iIz2w36XiMjN7vtfi8gXvqxbjc/pLCLZIhJc3ViN8cYSQQPgHiRKXsUikusx/euq7EtVL1bVN2t63bpIRO4XkW/KmR8rIvkicpqv+1LV2ap6YQ3FdVziUtWdqhqlqkU1sf9yPk9EJFlE1vtj/6bus0TQALgHiShVjQJ2Apd6zJtdsp6IhAQuyjrpbWC4iHQtM/9a4GdVXRuAmALhHKA10E1EBtXmB9u/ybrBEkEDJiLniUiKiPxRRPYBr4tISxH5TERSReSQ+76jxzaezR2TReQ7EZnmrrtNRC6u5rpdReQbEckSkYUi8oKIvF1B3L7E+BcR+d7d3xciEuux/DoR2SEiaSLyfxX9PqqaAiwGriuz6HpgVmVxlIl5soh85zF9gYhsFJEMEXkeEI9l3UVksRvfQRGZLSIt3GVvAZ2BT90ruj+ISLyIaMlBU0Tai8hcEUkXkSQRucVj34+IyHsiMsv9bdaJSEJFv4FrEvAfYJ773vN7nSoiX7qftV9EHnDnB4vIAyKy1f2clSLSqWys7rpl/518LyL/FJE04BFvv4e7TScR+cj9e0gTkedFJMyN6XSP9VqLSI6IxFXyfU0ZlggavrZADNAFmILzd/66O90ZyAWe97L9EGATEAs8CcwUEanGuv8GlgOtgEc48eDryZcYfwXcgHMmGwbcCyAifYHp7v7bu59X7sHb9aZnLCLSC+jvxlvV36pkH7HAR8CDOL/FVmCE5yrAX934+gCdcH4TVPU6jr+qe7Kcj5gDpLjbXwn8PxE532P5OHedFsBcbzGLSIS7j9nu61oRCXOXRQMLgfnuZ50CLHI3vQeYAPwCaAbcCOR4/WGOGQIkA22AJ7z9HuL0i3wG7ADigQ7AHFXNd7/jRI/9TgAWqWqqj3GYEqpqrwb0ArYDo9335wH5QLiX9fsDhzymlwA3u+8nA0keyyIABdpWZV2cg2ghEOGx/G3gbR+/U3kxPugx/Vtgvvv+YZwDRcmySPc3GF3BviOATGC4O/0E8J9q/lbfue+vB37wWE9wDtw3V7DfXwKry/s7dKfj3d8yBOcgWQREeyz/K/CG+/4RYKHHsr5ArpffdiKQ6u47HMgALnOXTfCMq8x2m4Dx5cwvjdXL77Szkr/v0t8DGFYSXznrDcFJmuJOJwJXB/L/X3192RVBw5eqqnklEyISISIvu00nmcA3QAupeETKvpI3qlpyxhdVxXXbA+ke8wB2VRSwjzHu83if4xFTe899q+oRIK2iz3Jjeh+43r16+TUwqwpxlKdsDOo5LSJtRGSOiOx29/s2zpWDL0p+yyyPeTtwzpRLlP1twqXitvhJwHuqWuj+O/mQY81DnXCuZsrjbVlljvu7r+T36ATsUNXCsjtR1R9xvt95ItIb54plbjVjatQsETR8ZcvL/h7oBQxR1WY4HYXg0YbtB3uBGLcZokQnL+ufTIx7PfftfmarSrZ5E7gauACIBj49yTjKxiAc/33/H87fy+nufieW2ae3ksB7cH7LaI95nYHdlcR0Are/43xgoojsE6cf6UrgF27z1i6gWwWb7wK6lzP/iPun59912zLrlP1+3n6PXUBnL4nsTXf964APPE96jO8sETQ+0Tht3YdFJAb4s78/UFV34Fy2P+J28g0DLvVTjB8AY0XkLLet+zEq/3f+LXAYmMGx9ueTieNz4FQRudw9gN3J8QfDaCAbyBCRDsB9ZbbfTwUHYFXdBSwF/ioi4SJyBnATzll0VV0HbMZJdv3dV0+cZqwJOG3z7URkqog0EZFoERnibvsq8BcR6SGOM0SklTrt87txkkuwiNxI+QnDk7ffYzlOYv2biES639mzv+Vt4DKcZDCrGr+BwRJBY/QM0BQ4CPyA0xFYG36N096bBjwOvAscrWDdaseoquuA23E6e/cCh3AObN62UZyDSBeOP5hUKw5VPQhcBfwN5/v2AL73WOVR4Eyc9vjPcTqWPf0VeFBEDovIveV8xASctvg9wMfAn1V1oS+xlTEJeFFV93m+gJeASW7z0wU4SXsfsAUY6W77NPAe8AVOH8tMnN8K4Bacg3kacCpO4vKmwt9DnXsnLsVp9tmJ83d5jcfyXcAqnCuKb6v+Exg41sliTK0SkXeBjarq9ysS07CJyGvAHlV9MNCx1FeWCEytEOdGpXRgG3Ah8AkwTFVXBzQwU6+JSDywBhigqtsCG039ZU1Dpra0xRlGmA08B9xmScCcDBH5C7AWeMqSwMmxKwJjjGnk7IrAGGMauXpX8Ck2Nlbj4+MDHYYxxtQrK1euPKiq5dZh8lsicHvyxwIHVPWEcr7uTTbP4tQqyQEmq+qqyvYbHx9PYmJiTYdrjDENmojsqGiZP5uG3gDGeFl+Mc746h44xdCm+zEWY4wxFfBbIlDVb3CGC1ZkPDBLHT/g1HBp5694jDHGlC+QncUdOL74VArHF84qJSJTRCRRRBJTU63CrDHG1KR6MWpIVWeoaoKqJsTF2TMnjDGmJgUyEezm+IqMHalGBUVjjDEnJ5CJYC5uDXgRGQpkqOreAMZjjDGNkj+Hj76D84SsWBFJwSnhGwqgqi/hPB/1F0ASzvDRG/wVizHGmIr5LRGo6oRKlitOuWBjjDFAYVExGbkFZOQWcDi3gIycAg7n5nM4x5l3fu/WnNGxRY1/br27s9gYY+oyVSWvoNg9mDsHcedAnu/Myzn+IF8yLyOngKyjJzyR8zixUU0sERhjjL8UFyv5RcUUFBVTUKQUFBWTX+hMZx8tPHYAzy0gIye/dPpwTgGZngf93ALyC4sr/JyQIKFFRCjNmzqv1tHh9GwdTfOIUFo0DTu2LCKUFk1DaRERRoumoUSHhxAS7J9uXUsExhi/KSp2D6hFxRQUHjvAHnewLbPs2MHYY313edllJQfq0unSdU9cv9BzurDsZylFxVWrxBwRFlx6MG8REUq32KgyB3HnoN7CnW7uHtQjw4JxKuzUHZYIjDGoKqlZR1m/N5P1ezNJOZRb4QG1oNDLAbZIKSg8doCt4rHVZyFBQmhwEKHBQlhIkPs+iJBgISw4qHRZaHAQUU1CjpsuXR5SZrpkXpC7rrvfMHcfzcsc1JuEBPvnywWAJQJjGpnComKSDx5h/Z5MNrgH/g17MzmYnV+6TkxkGOEhQaUHw5AgzwOuEB0a4nFAdQ/I5R5QnffHLStz8C7dNiSo9AAf5vG5TUKO/5zQoCCCgurWGXV9Z4nAmAYsM6+ADccd8LPYtD+rtA07LCSInm2iOL93a/q0a0bfds3o3a4ZzZuGBjhyU5ssERjTAKgqKYdyWVfmLD/lUG7pOq0iw+jbvhmTh8fTt10z+rRrRre4SEL91AFp6g9LBMbUM3kFRWzen+Uc8Pc4Z/kb9maWDj0UgW6xkfTv1IJfDelMn3bNOLVdM+Kim9S5TkpTN1giMKYOS806etwZ/vo9mSQfPFI6wiUyLJje7ZrxywEdnKad9s3o1SaapmENpyPT+J8lAmPqgJKmndW7DrN+z7EDf2rW0dJ12jcPp2/7Zow5rW1p007nmAjrODUnzRKBMQFQWFTMhr1ZJO5IJ3H7IRJ3pLM/0znohwYLPVpHc06POPq2b0afdtH0bdeMFhFhAY7aNFSWCIypBdlHC1m98xArth9i5Y50Vu88TE5+EQAdWjRlSNdWDIpvyYDOLenZJpqwEOvANbXHEoExfrDncC6JOw6xcns6K7YfYuO+TIoVggR6t23GVQM7MjA+hoQuLWnfommgwzWNnCUCY05SUbGyad+xZp6VOw6x+7AzbDMiLJgBnVtwx/k9GBTfkv6dWhAdbmP0Td1iicCYKsrJL2TNzsMk7jhE4o5DrN5xqHToZptmTUjoEsPNZ3cloUsMfdpF+61QmDE1xRKBMZU4kJlH4o5DrNiezsodh1i3J5OiYkUEeraOZlz/9iTEtyShSwwdWza1sfqm3vFrIhCRMcCzQDDwqqr+rczyLsBrQByQDkxU1RR/xmSMN8XFSlJqtnPQ3+6c8e9MzwGgSUgQ/Tu14DfndiOhSwxndm5J8whr5jH1n8+JQEQigTxVLfJx/WDgBeACIAVYISJzVXW9x2rTgFmq+qaInA/8FbjO5+iNqQFHC4uYtXQHy5LTWLnjEBm5BQDERoUxsEtLrh/WhYFdWnJq++Y2msc0SBUmAhEJAq4Ffg0MAo4CTUTkIPA58LKqJnnZ92AgSVWT3f3NAcYDnomgL3CP+/4r4JNqfg9jquXI0UJufWsl3yUdpHtcJGNObes088THEN8qwpp5TKPg7YrgK2Ah8CdgraoWA4hIDDAS+LuIfKyqb1ewfQdgl8d0CjCkzDr/Ay7HaT66DIgWkVaqmua5kohMAaYAdO7c2ZfvZUylDh3J54Y3VvBTymGeuvIMrkroFOiQjAkIb4lgtKoWlJ2pqunAh8CHInKyDaT3As+LyGTgG2A3cELTk6rOAGYAJCQk+OlRF6Yx2Z+Zx3Uzf2T7wRymTxzIRae2DXRIxgRMhYnAMwm47f1tPNdX1Z3lJQoPuwHPU6yO7jzPz9iDc0WAiEQBV6jq4ap8AWOqavvBI0yc+SOHjuTzxg2DGH5KbKBDMiagKu0sFpHfAX8G9gMlT2RW4IxKNl0B9BCRrjgJ4FrgV2X2HQuku81Of8IZQWSM32zYm8l1M5dTVFzMv28ZSr9OLQIdkjEB58uoobuAXmXb7SujqoUicgewAGf46Guquk5EHgMSVXUucB7wVxFRnKah26sUvTFVkLg9nRveWEFkWAhzpgzjlNbRgQ7JmDrBl0SwC8iozs5VdR4wr8y8hz3efwB8UJ19G1MVSzYd4Ddvr6Rd86a8ddNgOraMCHRIxtQZviSCZGCJiHyOM4QUAFV92m9RGVOD5v5vD/e8u4aebaKZddNgYqOaBDokY+oUXxLBTvcV5r6MqTfe/mEHD/1nLYO6xPDq5ASaWcE3Y05QaSJQ1UehdFQPqprt76CMOVmqyotLtvLUgk2c37s1L/76TMJD7fGNxpTHl1FDpwFvATHu9EHgelVd5+fYjKkWVeWJzzfw6nfbGN+/PdOu6keoVQA1pkK+NA3NAO5R1a8AROQ84BVguB/jMqZaCouK+dNHP/P+yhQmDevCny891Z7pa0wlfEkEkSVJAEBVl7gF6IypU/IKirhrzmoWrNvPnaN6cPfoHlYryBgf+DRqSEQewmkeApiIM5LImDoj+2ghU2YlsnRrGg+P7cuNZ3UNdEjG1Bu+NJzeiPO8gI/cV5w7z5g64dCRfH79yg/8uC2df1zVz5KAMVXky6ihQ8CdtRCLMVW2NyOX62YuZ2d6Di9NHMgFfdsEOiRj6h1vzyN4RlWnisinOLWFjqOq4/wamTGV2HbwCBNf/ZGM3ALevGEww7q3CnRIxtRL3q4ISvoEptVGIMZUxbo9GUx6bTnFCu/cMpTTOzYPdEjG1FveylCvdN/2V9VnPZeJyF3A1/4MzJiKLN+Wzk1vrCA6PIRZNw3hlNZRgQ7JmHrNl87iSeXMm1zDcRjjk8Ub93PdzB+Ja9aE928bbknAmBrgrY9gAs7zA7qKyFyPRdFAur8DM6as/6zZze/f+x+920Xz5g2DaWXF44ypEd76CJYCe4FY4B8e87OAn/wZlDFlvbVsOw/PXcfg+BhenZRAtBWPM6bGeOsj2AHsAIbVXjjGHE9VeX5xEv/4cjOj+7Th+V8NsOJxxtSwSvsIRGSoiKwQkWwRyReRIhHJ9GXnIjJGRDaJSJKI3F/O8s4i8pWIrBaRn0TkF9X5EqZhKi5W/vLZBv7x5WYuP7MDL020CqLG+IMvncXPAxOALUBT4Gbghco2ch94/wJwMdAXmCAifcus9iDwnqoOwHmm8Yu+h24assKiYu774Cde+34bk4fHM+3KfoRYBVFj/MKn/1mqmgQEq2qRqr4OjPFhs8FAkqomq2o+MAcYX3bXQDP3fXNgj29hm4Ysr6CI22av4sNVKdw9uid/vrSvVRA1xo98KTqXIyJhwBoReRKnA9mXBNIB53nHJVKAIWXWeQT4QkR+B0QCo8vbkYhMAaYAdO7c2YePNvVVVl4BU2atZFlyGo+OO5VJw+MDHZIxDZ4vB/TrgGDgDuAI0Am4ooY+fwLwhqp2BH4BvCUiJ8SkqjNUNUFVE+Li4mroo01dk5Z9lF+98iPLt6fzzDX9LQkYU0t8KTq3w32bCzxahX3vxkkaJTq68zzdhNvMpKrLRCQcZ7jqgSp8jmkA9hzO5bqZP5JyKJcZ1w1kVB8rHmdMbfF2Q9nPlFNsroSqnlHJvlcAPUSkK04CuBbnBjVPO4FRwBsi0gcIB1J9iNs0IMmp2Vw3czmZuQXMunEwQ7pZ8ThjapO3K4Kx7p+3u396PpimwgRRQlULReQOYAFO09JrqrpORB4DElV1LvB74BURudvd52RVrXTfpuFYu9spHgfwzpShnNbBiscZU9uksuOuiKx2h3d6zlulqmf6NbIKJCQkaGJiYiA+2tSw9CP5jJy2hKgmIbx102C6xVndIGP8RURWqmpCect86SwWERnhMTHcx+2M8erFr5LIyivg9RsGWRIwJoB8GT56E/CaiDQHBDiEParSnKTdh3OZ9cMOrjizIz3bRAc6HGMaNV9GDa0E+rmJAFXN8HtUpsF7duFmUJh6Qc9Ah2JMo+dt1NBEVX1bRO4pMx8AVX3az7GZBmrL/iw+WJnCDSO60qFF00CHY0yj5+2KINL9067bTY2a9sUmIsJCuH3kKYEOxRiD9zLUL7t/VuUmMmO8Wr3zEAvW7eeeC3oSExkW6HCMMXhvGnrO24aqemfNh2MaMlXl7/M30ioyjJvO6hrocIwxLm9NQyu9LDOmyr7ZcpAfktN55NK+RDbxZcCaMaY2eGsaerM2AzENW3Gx8uT8jXRs2ZRfDekS6HCMMR4qPS0TkTjgjzgPlwkvma+q5/sxLtPAfP7zXtbtyeSf1/QjLMTuRzSmLvHlf+RsYAPQFaf66HacgnLG+KSgqJh/fLGJ3m2jGdevQ6DDMcaU4UsiaKWqM4ECVf1aVW8E7GrA+OzdFbvYnpbDfRf1ItieNGZMneNLj12B++deEbkE53GSMf4LyTQkuflFPLtoC4PiW3J+79aBDscYUw5vw0dDVbUAeNwtL/F74F84zxi+u5biM/Xc60u3kZp1lBd/fWbpXenGmLrF2xXBbhGZC7wDZKrqWmBk7YRlGoLDOflMX7KVUb1bMyjeLiKNqau89RH0wekUfhDYJSLPisjQ2gnLNATTv95K9tFC7hvTK9ChGGO8qDARqGqaqr6sqiOBwUAy8E8R2SoiT/iycxEZIyKbRCRJRO4vZ/k/RWSN+9osIoer/U1MnbIvI483vt/OZf070Ltts0CHY4zxwqcB3aq6B5gJTAeygJsr20ZEgoEXgItx7kGYICJ9y+z3blXtr6r9cfofPqpa+KauenbRFopVudvKTBtT53lNBCISLiJXichHQBLOsNH7gfY+7HswkKSqyaqaD8wBxntZfwJOf4Sp57amZvNe4i5+PaQLnWIiAh2OMaYS3kYN/RsYDXyNc1PZr1Q1rwr77gDs8phOAYZU8FldcG5YW1zB8inAFIDOnTtXIQQTCE9/sZkmIUFWZtqYesLbFcF8oLuqXqWqH1YxCVTVtcAHqlpU3kJVnaGqCaqaEBcX58cwzMn6KeUwn/+8l5vP7kZcdJNAh2OM8YG3zuJZqpp1EvveDXTymO7ozivPtVizUIPw1IJNtIwI5Zazrcy0MfWFP6t/rQB6iEhXEQnDOdjPLbuSiPQGWgLL/BiLqQXfJx3k2y0HuX3kKUSHhwY6HGOMj/yWCFS1ELgDWIBTtO49VV0nIo+JyDiPVa8F5qiq+isW438lD51p3zyciUOtzLQx9Ym3zuLLvW2oqpUO9VTVecC8MvMeLjP9SGX7MXXf/LX7+Cklg6euPIPw0OBAh2OMqQJvJSYudf9sDQzn2IiekcBSbMy/cRUWFfPUF5vo0TqKy8/sGOhwjDFV5O0JZTcAiMgXQF9V3etOtwPeqJXoTL3wwcoUklOP8PJ1A63MtDH1kC99BJ1KkoBrP2CD+Q0AeQVFPLNwCwM6t+DCvm0CHY4xphp8eR7BIhFZwLHhndcAC/0XkqlPZi3bzr7MPJ65tr+VmTamnqo0EajqHW7H8dnurBmq+rF/wzL1QUZuAS98tZVze8YxtFurQIdjjKkmX64ISkYIWeewOc6Mb7aSkVvAfRdZmWlj6rNK+whE5HIR2SIiGSKSKSJZIpJZG8GZuutAZh6vfbedcf3ac1qH5oEOxxhzEny5IngSuFRVN/g7GFN//GtxEgVFxdxjZaaNqfd8GTW035KA8bQj7QjvLN/JtYM7ER8bGehwjDEnyZcrgkQReRf4BDhaMtOXO4tNw/SPLzYTGhzEnef3CHQoxpga4EsiaAbkABd6zFOs87hRWrcng7n/28PtI7vTull4oMMxxtQAX4aP3lAbgZj64akFm2jeNJQp53QPdCjGmBpSaSIQkXDgJuBUoPQUUFVv9GNcpg76ITmNJZtS+dPFvWne1MpMG9NQ+NJZ/BbQFrgI57GVHXEeYG8akZIy022bhTNpeHygwzHG1CBfEsEpqvoQcERV3wQuoYJnD5uG68v1+1m98zBTR/ewMtPGNDC+JIIC98/DInIa0BynNLVpJIqKlacWbKJbbCRXDrQy08Y0NL4kghki0hJ4EOdRk+uBv/uycxEZIyKbRCRJRO6vYJ2rRWS9iKwTkX/7HLmpNR+v3s2WA9nce1EvQoL9+XRTY0wg+DJq6FX37TdAN193LCLBwAvABUAKsEJE5qrqeo91egB/Akao6iERsSuNOiavoIh/frmZMzo25+LT2gY6HGOMH/jz9G4wkKSqyaqaD8wBxpdZ5xbgBVU9BKCqB/wYj6mG2T/uZPfhXP44preVmTamgfJnIugA7PKYTnHneeoJ9BSR70XkBxEZU96ORGSKiCSKSGJqaqqfwjVlZeUV8MJXSZx1SiwjTokNdDjGGD8JdINvCNADOA+YALwiIi3KrqSqM1Q1QVUT4uLiajnExuvVb7eRfiTfykwb08D5ckPZ5eXMzgB+rqQpZzfQyWO6ozvPUwrwo6oWANtEZDNOYlhRWVzGvw5mH+XVb5P5xelt6dfphNxsjGlAfKk1dBMwDPjKnT4PWAl0FZHHVPWtCrZbAfQQka44CeBa4Fdl1vkE50rgdRGJxWkqSq7SNzB+8fziJPIKi/n9hXY1YExD50siCAH6qOp+ABFpA8zCuansG5w7j0+gqoUicgewAAgGXlPVdSLyGJCoqnPdZReKyHqgCLhPVdNO9kuZk7MrPYfZP+7g6oSOdI+LCnQ4xhg/8yURdCpJAq4D7rx0ESmoaCMAVZ0HzCsz72GP9wrc475MHfHPhZsJEuGuUfbQGWMaA18SwRIR+Qx4352+wp0XCRz2W2QmIDbuy+Tj1buZck432ja3MtPGNAa+JILbcQ7+I9zpWcCH7tn8SH8FZgJj2oJNRDUJ4bZzrcy0MY2FL3cWK/CB+zINWOL2dBZuOMB9F/WiRURYoMMxxtSSSu8jEJHLRWSLiGSISKaIZIlIZm0EZ2pPSZnpuOgm3DAiPtDhGGNqkS83lD0JjFPV5qraTFWjVbWZvwMzteurTQdYsf0Qd43qQUSYLy2GxpiGwpdEsF9VN/g9EhMwxcXKk/M30aVVBNcM6lT5BsaYBsWXU79EEXkX5+avoyUzVdUeXt9AzP3fHjbuy+K5CQMItTLTxjQ6viSCZkAOcKHHPAUsETQA+YXF/OPLTZzavhljT28X6HCMMQHgy6ihG2ojEBMY7yzfya70XN688XSCgqzMtDGNUYWJQET+oKpPisi/cK4AjqOqd/o1MuN3R44W8q/FWxjaLYZzeliZaWMaK29XBCUdxIm1EYipfa99t42D2fnMuN4eOmNMY1ZhIlDVT90/36y9cExtST+Sz8vfJHNh3zac2blloMMxxgSQL88j6AncC8R7rq+q5/svLONvL36VRE5+oT10xhjj06ih94GXgFdxSkWbei7lUA6zlu3gijM70qNNdKDDMcYEmC+JoFBVp/s9ElNr/vnlFhC4+wIrM22M8e3O4k9F5Lci0k5EYkpefo/M+MXGfZl8tDqFG4bH075F00CHY4ypA3xJBJOA+4ClOI+oXImPI4lEZIyIbBKRJBG5v5zlk0UkVUTWuK+bqxK8qbon528iukkIt51nZaaNMQ5fbijrWp0di0gw8AJwAc5D6leIyFxVXV9m1XdV9Y7qfIapmh+S01i88QD3X9zbykwbY0p5u6HsfFVdLCKXl7fch1pDg4EkVU129zcHGA+UTQSmFqgqf/vvRto2C2fy8PhAh2OMqUO8XRGcCywGLi1nmS+1hjoAuzymU3AeeF/WFSJyDrAZuFtVd5VdQUSmAFMAOnfuXMnHmvIsWLePNbsO8+QVZxAeGhzocIwxdYi3G8r+7P7pz1pDnwLvqOpREbkVeBM44f4EVZ0BzABISEg4odyF8a6wqJgn52+iR+soLj+zQ6DDMcbUMT49gURELgFOBUqfZq6qj1Wy2W7As7h9R3deKVVN85h8FechOKaGvZeYQvLBI7xyfQIhVmbaGFOGL4+qfAm4BvgdIMBVQBcf9r0C6CEiXUUkDLgWmFtm3551j8dxrL6RqSE5+YU8s3AzCV1aMrpP60CHY4ypg3w5PRyuqtcDh9hSWfEAACAASURBVFT1UWAYUOmdSKpaCNwBLMA5wL+nqutE5DERGeeudqeIrBOR/wF3ApOr8yVMxV7/fjsHso7yx4utsJwxpny+NA3luX/miEh7IA3w6QkmqjoPmFdm3sMe7/8E/Mm3UE1VpR/J56UlWxndpw2D4u0eQGNM+XxJBJ+KSAvgKWAVzoihV/walakRL3yVxJH8Qv4wxgrLGWMq5jURiEgQsEhVDwMfishnQLiqZtRKdKbadqXn8NayHVw5sCM9rbCcMcYLr30EqlqMc3dwyfRRSwL1wz+/3IwITB1theWMMd750lm8SESuEOtprDfW78nk4zW7mTzCCssZYypXYSIQkf/nvr0V55kER0UkU0SyRCSzVqIz1fLkgo1ENwnht+eeEuhQjDH1gLcrgjEAqhqtqkGqGqaqzdzpZrUUn6miZVvTWLIpldtHnkLziNBAh2OMqQe8dRYHi0hLnJvITqCq6f4JyVSXqvK3+Rtp1zycSVZYzhjjI2+JoDfOswfKSwQKdPNLRKba5q/dx/92HebJK62wnDHGd94SwXpVHVBrkZiTUlBUzFMLNtGzTRRXnNkx0OEYY+oRq0DWQLyXuIvkg0f4w0W9CQ6yAV7GGN95SwTP1loU5qQ4heW2MCi+JaOssJwxpoq8JYKzROT08haISKSI3Cgiv/ZTXKYKXvtuG6lZR7nfCssZY6rBWx/B88BDbjJYC6TiPI+gB9AMeA2Y7fcIjVfpR/J56etkLuzbhoFdrLCcMabqvD2hbA1wtYhEAQk4FUdzgQ2quqmW4jOVeH5xEjlWWM4YcxJ8qT46EvjcrTtk6pBd6Tm89cN2rk7oxCmtrbCcMaZ6fBk1dA2wRUSeFJHe/g7I+O7pLzcTJGKF5YwxJ6XSRKCqE4EBwFbgDRFZJiJTRKTSU1ARGSMim0QkSUTu97LeFSKiIpJQpegbsXV7MvhkzW5uPKsrbZuHV76BMcZUwKf7CFQ1E/gAmIPTV3AZsEpEflfRNiISjFPC+mKgLzBBRPqWs140cBfwY5Wjb8SenL+JZuGh/Obc7oEOxRhTz/ny8PpxIvIxsAQIBQar6sVAP+D3XjYdDCSparKq5uMkkfHlrPcX4O8ceySmqcTSpIN8vTmV20d2p3lTKyxnjDk5vlwRXAH8U1VPV9WnVPUAgKrmADd52a4DsMtjOsWdV0pEzgQ6qernVQu78SopLNe+eTjXD4sPdDjGmAbAl0TwCLC8ZEJEmopIPICqLqruB7uPwXwa71cVJetOEZFEEUlMTU2t7kc2CPN+3sdPKRncfUFPKyxnjKkRviSC9wHPoaNF7rzK7AY6eUx3dOeViAZOA5aIyHZgKDC3vA5jVZ2hqgmqmhAXF+fDRzdMTmG5jfRsE8XlVljOGFNDfEkEIW4bPwDu+zAftlsB9BCRriISBlwLzPXYT4aqxqpqvKrGAz8A41Q1sUrfoBGZs2IX29Ny+OMYKyxnjKk5viSCVBEZVzIhIuOBg5VtpKqFwB3AAmAD8J6qrhORxzz3Z3xz5Gghzy7cwuD4GM7vbYXljDE1x5c7i38DzBaR53EeUrMLuN6XnavqPGBemXkPV7Dueb7ss7Ga+d02DmYf5eXrBlphOWNMjao0EajqVmCoW3MIVc32e1TmOGnZR3n5661cdGobBnZpGehwjDENjC9XBIjIJcCpQHjJ2aiqPubHuIyHfy1OIregiPsusgofxpia58sNZS/h1Bv6HU7T0FVAFz/HZVw703KY/eMOrhnUiVNaRwU6HGNMA+RLZ/FwVb0eOKSqjwLDAKtyVkv+8eUmgoOEu0bZT26M8Q9fEkFJ6YccEWkPFODUGzJ+tnZ3Bv9Zs4cbR1hhOWOM//jSR/CpiLQAngJWAQq84teoDAB/n7+RFhGh3GqF5YwxfuQ1EbhlIBap6mHgQxH5DAhX1Yxaia4R+27LQb7dcpAHL+ljheWMMX7ltWnIfSrZCx7TRy0J+F9xsfL3+Rvp0KIpE4dav7wxxr986SNY5D44xu5iqiWf/7yXn3dncI8VljPG1AJfEsGtOEXmjopIpohkiUimn+NqtAqKipn2xSZ6t43mlwM6VL6BMcacJF/uLLanoteiOct3siMth9cnD7LCcsaYWlFpIhCRc8qbr6rf1Hw4jduRo4U8u2gLQ7rGcF6vxltu2xhTu3wZPnqfx/twnEdQrgTO90tEjdir327jYHY+r1zf2wrLGWNqjS9NQ5d6TotIJ+AZv0XUSB3MPsqMb7Zy8WltGdDZCssZY2qPL53FZaUAfWo6kMbu+cVJ5BUWc+9FvQIdijGmkfGlj+BfOHcTg5M4+uPcYWxqyI60I8z+cQdXJ3Sie5wVljPG1C5f+gg8Hx1ZCLyjqt/7KZ5G6R9fbCY4SJg6ukegQzHGNEK+JIIPgDxVLQIQkWARiVDVnMo2FJExwLNAMPCqqv6tzPLfALcDRUA2MEVV11fxO9Rra3dnMPd/e7h9ZHfaNLPCcsaY2ufTncVAU4/ppsDCyjYSkWCc8hQXA32BCSLSt8xq/1bV01W1P/Ak8LRPUTcgVljOGBNoviSCcM/HU7rvI3zYbjCQpKrJqpoPzAHGe66gqp53KEdyrC+iUfh2SyrfbjnIHSNPoVm4FZYzxgSGL01DR0TkTFVdBSAiA4FcH7brgPOg+xIpwJCyK4nI7cA9QBgV3JsgIlOAKQCdO3f24aPrPs/CctcNs8JypuoKCgpISUkhLy+v8pVNoxEeHk7Hjh0JDfX95NKXRDAVeF9E9uA8qrItzqMra4SqvgC8ICK/Ah4EJpWzzgxgBkBCQkKDuGr47Oe9rN2dydNX96NJiBWWM1WXkpJCdHQ08fHxdgOiAUBVSUtLIyUlha5du/q8nS83lK0Qkd5AyQD3Tapa4MO+dwOdPKY7uvMqMgeY7sN+6738wmKmLXAKy43vb4XlTPXk5eVZEjDHERFatWpFampqlbbz5eH1twORqrpWVdcCUSLyWx/2vQLoISJdRSQMuBaYW2bfnuMlLwG2+B56/fXO8p3sTM/hjxf3tsJy5qRYEjBlVeffhC+dxbe4TygDQFUPAbdUtpGqFgJ3AAuADcB7qrpORB4TkXHuaneIyDoRWYPTT3BCs1BDk320kOcWbWFotxjO62mF5YwxgedLH0GwiIiqKpQOCw3zZeeqOg+YV2bewx7v76pCrPWWqrI19QjLth5k3s/7SDuSz8yL+9jZnKnX0tLSGDVqFAD79u0jODiYuDjn5Gb58uWEhVV8mEhMTGTWrFk899xzXj9j+PDhLF26tMZinjp1Ku+//z67du0iKKg6FXYaJl8SwXzgXRF52Z2+1Z1nKqCq7EzPYenWNJZtTWNZchqpWUcBaN88nPsu6kX/Ti0CHKUxJ6dVq1asWbMGgEceeYSoqCjuvffe0uWFhYWEhJR/iElISCAhIaHSz6jJJFBcXMzHH39Mp06d+Prrrxk5cmSN7duTt+9dV/kS7R9xhm7e5k5/Cbzit4jqqd2Hc1m2NY2lWw/yw9Y09mQ4Q/riopswrFsrhndvxbDuregcE2FXAqbGPfrpOtbvqdkHB/Zt34w/X3pqlbaZPHky4eHhrF69mhEjRnDttddy1113kZeXR9OmTXn99dfp1asXS5YsYdq0aXz22Wc88sgj7Ny5k+TkZHbu3MnUqVO58847AYiKiiI7O5slS5bwyCOPEBsby9q1axk4cCBvv/02IsK8efO45557iIyMZMSIESQnJ/PZZ5+dENuSJUs49dRTueaaa3jnnXdKE8H+/fv5zW9+Q3JyMgDTp09n+PDhzJo1i2nTpiEinHHGGbz11ltMnjyZsWPHcuWVV54Q30MPPUTLli3ZuHEjmzdv5pe//CW7du0iLy+Pu+66iylTpgAwf/58HnjgAYqKioiNjeXLL7+kV69eLF26lLi4OIqLi+nZsyfLli0rvcLyN19GDRUDL7kvRORs4F84pSEarQOZeSxLTnMP/mnsTHcqbsREhjG0Wwy3dY9lWLdWdI+LtAO/aVRSUlJYunQpwcHBZGZm8u233xISEsLChQt54IEH+PDDD0/YZuPGjXz11VdkZWXRq1cvbrvtthPGwa9evZp169bRvn17RowYwffff09CQgK33nor33zzDV27dmXChAkVxvXOO+8wYcIExo8fzwMPPEBBQQGhoaHceeednHvuuXz88ccUFRWRnZ3NunXrePzxx1m6dCmxsbGkp6dX+r1XrVrF2rVrS4dtvvbaa8TExJCbm8ugQYO44oorKC4u5pZbbimNNz09naCgICZOnMjs2bOZOnUqCxcupF+/frWWBMC3KwJEZAAwAbga2AZ85M+g6qK07KP8kJzOsuSDLNuaxtbUIwA0Cw9hSLdWTB4ez7DurejVJpogGwlkallVz9z96aqrriI42Lk3JiMjg0mTJrFlyxZEhIKC8keeX3LJJTRp0oQmTZrQunVr9u/fT8eOHY9bZ/DgwaXz+vfvz/bt24mKiqJbt26lB98JEyYwY8aME/afn5/PvHnzePrpp4mOjmbIkCEsWLCAsWPHsnjxYmbNmgVAcHAwzZs3Z9asWVx11VXExsYCEBMTU+n3Hjx48HFj95977jk+/vhjAHbt2sWWLVtITU3lnHPOKV2vZL833ngj48ePZ+rUqbz22mvccMMNlX5eTaowEYhIT5yD/wTgIPAuIKrqn4a1OiYjp4Aftjln/D8kp7FxXxYAkWHBDO4awzWDOjGsWyx92zezIaDGeIiMjCx9/9BDDzFy5Eg+/vhjtm/fznnnnVfuNk2aNCl9HxwcTGFhYbXWqciCBQs4fPgwp59+OgA5OTk0bdqUsWPH+rwPgJCQEIqLiwGnzyE/P790mef3XrJkCQsXLmTZsmVERERw3nnneb0DvFOnTrRp04bFixezfPlyZs+eXaW4Tpa3K4KNwLfAWFVNAhCRu2slqgDIPlrIim3pLN16kGXJaazbk4kqhIcGkdAlhvsuas+w7q04vUNzQoPr8WiD7FTY/zPsWwtF+RDVxn21dl6RcRBsdY9MzcjIyKBDB+emyTfeeKPG99+rVy+Sk5PZvn078fHxvPvuu+Wu98477/Dqq6+WNh0dOXKErl27kpOTw6hRo5g+fTpTp04tbRo6//zzueyyy7jnnnto1aoV6enpxMTEEB8fz8qVK7n66quZO3duhVc4GRkZtGzZkoiICDZu3MgPP/wAwNChQ/ntb3/Ltm3bSpuGSq4Kbr75ZiZOnMh1111XekVVW7wlgstxbgL7SkTm49z522BOfXPzi0jckV7axv/z7gyKipWw4CAGdG7B1FE9Gda9Ff06Na+fJSCKiyBtK+z7CfavhX3uwT97X+XbRrRykkNknEeS8EgWJcmjaQzYEDzjxR/+8AcmTZrE448/ziWXXFLj+2/atCkvvvgiY8aMITIykkGDBp2wTk5ODvPnz+ell14qnRcZGclZZ53Fp59+yrPPPsuUKVOYOXMmwcHBTJ8+nWHDhvF///d/nHvuuQQHBzNgwADeeOMNbrnlFsaPH0+/fv1KP7M8Y8aM4aWXXqJPnz706tWLoUOHAhAXF8eMGTO4/PLLKS4upnXr1nz55ZcAjBs3jhtuuKHWm4XAaerxvoJIJE7V0Ak4ReFmAR+r6hf+D+9ECQkJmpiYWPmKZeQVFLF652G3g/cga3YdpqBICQkS+nVqUTqy58wuLQkPrWcH/qNZsH+de7D/2Tnw718PhW5twKBQiOsNbU+Htqc5f7Y5DUKbQvYB97Ufjni895yfvR8Ky7mslWA3WbQukyw8k4g7L7w5WKd5jdqwYQN9+thTY7Ozs4mKikJVuf322+nRowd3313/Gi8SExO5++67+fbbb096X+X92xCRlapa7phdX0YNHQH+DfxbRFoCV+EMKQ1IIqiu6Uu28uyiLQQJnN6hOTee1ZVh3VoxKD6GyCb1ZMyvKmSkeJzhu69D246t07Slc6BPuPHYgT+2F4RUcHNPyy7Oq7LPPZoFR1KPJYbsskljv5N8jhyA4nLaboPDKkgU5VxthJV/lmVMeV555RXefPNN8vPzGTBgALfeemugQ6qyv/3tb0yfPr3W+wZKVHpFUNdU94ogOTWb5NQjDO4WUz9q/xcehdRNx87wSw76eSXVPgRiunmc4Z/u/NmsfWDPvIuLnRhLE0aZ5OF51XHkIOU+giIsyu2vKNMUFRV3YjIJaXLi9o2EXRGYitT4FUFD0S0uim519cHwR9KOdeCWHPhTNx47sw6NgNZ94dTL3AP/Gc50kzr4fYKCICLGebWu5CBVVAg5B703T6Vugm3feCTAMsJbeOnHKEkmbSAyFoLqWZOfMbWk0SSCOiNjN6QsP9Z5u+9nyNpzbHl0O+fMvseFbtPO6c6Zf0M8iAWHQHRb51WZwqMnXlGUbZ7avdJpvsrPPnF7CYKI2AquLMokj6YtrT/DNCqWCPytIA92fA9bF0PSIkjd4MwPCnHa7ruefazztu3pzpmrOVFIE2jRyXlV5mi2R8Io049R0s9xcIszr+joidsHhR5/ZVG249szaYRFWdIw9Z4lgpqmCgc3Owf9rYtg+3fOiJvgJtBlGPT/lXPwb923Ubdv+1WTKOcV0837eqqQl+F91FTmHtizxkkgWnTiPkIjKh9mWzKyKrSpf76vMSfJEkFNyD0M2752Dv5JiyAzxZnfqgcMvAFOGQVdRkBYRGDjNMcTgaYtnFdcT+/rFhdDbrr3UVPpybBzGeSklb+PJs09kkQ5/RhRrZ2mwajW9eIqY+TIkdx///1cdNFFpfOeeeYZNm3axPTp5T9s8LzzzmPatGkkJCTwi1/8gn//+9+0aHF8Jd7yKpmW9cknn9CzZ0/69u0LwMMPP8w555zD6NGja+Cb1cFy1apQXOA0cQbV/GHbEkF1FBc5Z4lb3QN/ygrnbLFJM+h6Dpxzr3Pwb9E50JGamhIU5DTbRcZCm0rq+hQVuE1QXjrB962F7MVwNOPE7SNaeTQXnuEOAe5Z5+74njBhAnPmzDkuEcyZM4cnn3zSp+3nzZtX+UoV+OSTTxg7dmxpInjssceqva+yaq1cdUEBIUE4g0KKCpw/iwucQRSefxYXHhs40ryTX5qP/ZoIRGQM8CwQDLyqqn8rs/we4GagEEgFblTVHf6Mqdoy97rt/Ash+SvIPQQItB8AZ98D3UdBx4Q695/VBEBwqDOMt1n7ytctyHU7wd2+i8O73BsC18LyV471YQSHuTcFnnFsyLC2PLaf/97vDDyoSW1Ph4v/VuHiK6+8kgcffJD8/HzCwsLYvn07e/bs4eyzz+a2225jxYoV5ObmcuWVV/Loo4+esH18fDyJiYnExsbyxBNP8Oabb9K6dWs6derEwIEDAecegRkzZpCfn88pp5zCW2+9xZo1a5g7dy5ff/01jz/+OB9++CF/+ctfSstDL1q0iHvvvZfCwkIGDRrE9OnTadKkCfHx8UyaNIlPP/2UgoIC3n//fXr37n1CXCddrvrN15k8+QbGXnwhV/5yLBQVENW2K9m71rPkm+946K//pGXzaDZuSWbzd5/wyxvvYdeefeQdzeeumyYwZeKVEBzK/CXLeOCJZygqLia2VSu+/OwDevUfytKly2q8XLXfEoH7JLMXgAuAFGCFiMxV1fUeq60GElQ1R0RuA54ErvFXTFVSeNS5zE9aCEmL4cA6Z35UG+h5sXPG320kRLYKbJymfgttWvFNfUWFkJbkDil27yPZsgDWvO0sv+g92A+ENIX8I85ZowS5zUr+b1qKiYlh8ODB/Pe//2X8+PHMmTOHq6++GhHhiSeeICYmhqKiIkaNGsVPP/3EGWecUe5+Vq5cyZw5c1izZg2FhYWceeaZpYng8ssv55ZbnCfjPvjgg8ycOZPf/e53jBs37rjnApTIy8tj8uTJLFq0iJ49e3L99deX1hECiI2NZdWqVbz44otMmzaNV1999YR4yi1XHRzEnb+7g3NHDOXjf79OUUEe2RkZrFu2kMcf/TNLP3+H2JbRpB886N7vkwFZ++DQdmenJTdlahGrfl7P2qVf0LVrNwgK4bWZM4mJbU3u0QIGDT+bK26cSrEqt9z76PHlqqNjmDjxOr+Uq/bnFcFgIElVkwFEZA5OqYrSRKCqX3ms/wMw0Y/xeKfq1OZJWnisk7cgxxlB0mUYjH7UOfi3Oa1etN+aBiA4BFr3dl5cdWx+1n4nMWRFQ2ikU0pk0I3HlkuQk2BCmjp/lrz3Q1t3SfNQSSKYOXMmAO+99x4zZsygsLCQvXv3sn79+goTwbfffstll11GRITThzZu3LjSZWvXruXBBx/k8OHDZGdnH9cMVZ5NmzbRtWtXevZ0+nwmTZrECy+8UJoILr/8cgAGDhzIRx99BIX5x5pfigrIz8th3uef8vSDdxKdv48h/fuy4J2XGDv6LBYvWsisv98Lh7YRDDQPhlmLF3HVpRc65aqDQ4jp2ML5ewuLdE4a43o7xxAJcq7kWhxk8OAhdO13VmnMz0178fhy1UlJtV6u2p+JoAOwy2M6BRjiZf2bgP+Wt0BEpuA8JY3OnWuw3T0v81gn79ZFcHinMz+mOwyY6DT3xJ9VN2/cMo1XdBvntWEDxMQ784qLnYRQ4L4Kc53O7ZziY9uFhB+fHEKbnnRT5vjx47n77rtZtWoVOTk5DBw4kG3btjFt2jRWrFhBy5YtmTx5stcSzN5MnjyZTz75hH79+vHGG2+wZMkS7xuoOq+CHOeKKi/TubrPSIHiQppkp8CBTIIPJVOYc/jYlb5rwRdfc/hwBqefPQYQcnLzaBoZxdjLrnLu5YnpBk0jnYN9UAg0+w5ygiD2lOP2ExIeSXFIOIQ2rRflqutAdziIyEQgAXiqvOWqOkNVE1Q14aQug4qLYfcq+OYpeO1i+Hs8vDsRfn7fKdFwyT/gzjVw5yr4xVPQa4wlAVM/BAU5Z6GRsc69FrE9j92B3rIrRLV1hjAXHHFuYEzfeqx0SVoSZO6GnHQniVSh7ExUVBQjR47kxhtvLC3xnJmZSWRkJM2bN2f//v3897/lnt+VOuecc/jkk0/Izc0lKyuLTz/9tHRZVlYW7dq2peBoLrPffsvpVM1JJzo8mKwDO+HQDif+vAw4vINezfLYnpxE0vIvIX0rb735Oucm9HVGcpUkiZAm0CTa6Xdp3gladnN+r9Z9eWfBj7w68zW279zN9p0pbNu+gy+XfE9OcDNGjRrN9NdnQ1gERQSTkZnF+eefz/vvv09amjNSrORJZiXlqoFql6v+5ptv2LZt23H7hWPlqj0fAHSy/HlFsBvwvPunozvvOCIyGvg/4FxVLefunhqyahYsfOTY0L52/WDEXU5zT8fBFRdlM6a+EnEOeiFNnCGyJYoLj105lLyOpnKs7pO4/x98awKdcPFZXPbee8x54Qk4sIF+7cIY0KcbvXt0o1P7toxIOMO5H+PABudMPX0bHIh0mmRSN3Nmx5Zcc8lI+p3Wh9axrRh0ei+n4/zABv5y320MGXQmca1aMmTAaWRlZ8HhHVx78Vncct9feG76K3ww8zm3OSyC8NjOvP7y81z124coLCpiUEICv7nvUWga4Vz9xJ4CMbEQne4kAo8RODk5OcxfsICXXn65dF5jKVftt6JzIhICbAZG4SSAFcCvVHWdxzoDgA+AMaq6xZf9VrfoHFsWws/vOc093Uc6Y7WNqcdqtOicFjtNKCWJoSi/8m1qgwS5zTChTlNMcKjzPjjEKYPeCPvrfClXXWeKzqlqoYjcASzAGT76mqquE5HHgERVnYvTFBQFvO8+4H2nqo6rcKcno8do52WMOVFJB7Pd/Vyn+atcdaMpQ21MQ2NlqE1FqnpFUCc6i40x1VPfTuSM/1Xn34QlAmPqqfDwcNLS0iwZmFKqSlpaGuHh4VXazmoNGVNPdezYkZSUFFJTUwMdiqlDwsPD6dixY5W2sURgTD0VGhpaeuepMSfDmoaMMaaRs0RgjDGNnCUCY4xp5OrdfQQikgpU95kFscDBGgynvrPf43j2exxjv8XxGsLv0UVVyy3WVu8SwckQkcSKbqhojOz3OJ79HsfYb3G8hv57WNOQMcY0cpYIjDGmkWtsiWBGoAOoY+z3OJ79HsfYb3G8Bv17NKo+AmOMMSdqbFcExhhjyrBEYIwxjVyjSQQiMkZENolIkojcH+h4AkVEOonIVyKyXkTWichdgY6pLhCRYBFZLSKfBTqWQBORFiLygYhsFJENIjIs0DEFiojc7f4/WSsi74hI1cp61hONIhGISDDwAnAx0BeYICJ9AxtVwBQCv1fVvsBQ4PZG/Ft4ugvYEOgg6ohngfmq2hvoRyP9XUSkA3AnkKCqp+E8afHawEblH40iEQCDgSRVTVbVfGAOMD7AMQWEqu5V1VXu+yyc/+QdAhtVYIlIR+AS4NVAxxJoItIcOAeYCaCq+ap6OLBRBVQI0NR9BnsEsCfA8fhFY0kEHYBdHtMpNPKDH4CIxAMDgB8DG0nAPQP8ASgOdCB1QFcgFXjdbSp7VUQiAx1UIKjqbmAasBPYC2So6heBjco/GksiMGWISBTwITBVVTMDHU+giMhY4ICqrgx0LHVECHAmMF1VBwBHgEbZpyYiLXFaDroC7YFIEZkY2Kj8o7Ekgt1AJ4/pju68RklEQnGSwGxV/SjQ8QTYCGCciGzHaTI8X0TeDmxIAZUCpKhqyVXiBziJoTEaDWxT1VRVLQA+AoYHOCa/aCyJYAXQQ0S6ikgYTofP3ADHFBAiIjjtvxtU9elAxxNoqvonVe2oqvE4/y4Wq2qDPOvzharuA3aJSC931ihgfQBDCqSdwFARiXD/34yigXacN4pHVapqoYjcASzA6fl/TVXXBTisQBkBXAf8LCJr3HkPqOq8AMZk6pbfAbPdk6Zk4IYAxxMQqvqjiHwArMIZbbeaBlpqwkpMGGNMI9dYmoaMMcZUwBKBMcY0cpYIjDGmNnoVBwAAAfNJREFUkbNEYIwxjZwlAmOMaeQsERhTi0TkPKtwauoaSwTGGNPIWSIwphwiMlFElovIGhF52X1eQbaI/NOtT79IROLcdfuLyA8i8pOIfOzWqEFEThGRhSLyPxFZJSLd3d1HedT7n+3etWpMwFgiMKYMEfn/7d29ShxRGMbx/2MT4geEFDYpFDsJxISAjVjlBiwUQVm8gDTpQiAi5B4CsdyghQjaBywWrNRCELwCQdgmCCkSZHks5hSrplhF3WKeXzXzzpnDnGLmnQ/mPZPAIjBj+y3QAZaBIeDI9mugBayVXX4Cn22/AU664pvAd9tTVDVqzkv8HfCJam6MCaq/vSP6phYlJiLu6APwHjgsN+vPgTZVmeqt0mYD2Cn1+1/YbpV4E9iWNAK8sr0LYPsvQOnvwPZZWT8GxoH9xx9WxP8lEUTcJqBp+8u1oLR6o91967P861rukPMw+iyvhiJu2wPmJY0CSHopaYzqfJkvbZaAfdsXwG9JsyXeAFpl9rczSXOlj2eSBp90FBE9yp1IxA22TyV9BX5JGgAugY9Uk7RMl21tqu8IACvAj3Kh767W2QDWJX0rfSw84TAiepbqoxE9kvTH9nC/jyPioeXVUEREzeWJICKi5vJEEBFRc0kEERE1l0QQEVFzSQQRETWXRBARUXNX3unMwTtygBcAAAAASUVORK5CYII=\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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mpXsMERE0oqZLatrm5qhREQkJiULERGJSclCRERiysg+CxFpObt376a0tJQdO3bE3llSQm5uLoWFhbRp0ybuY5QsRKRJSktLyc/Pp3///tS9bpWkCndn/fr1lJaWMmBAzRV/66ZmKBFpkh07dtC1a1clijRhZnTt2rXBNUElCxFpMiWK9NKYfy8liyilG7dxy/OLWL5hW+ydRURaESWLKFt2VvDHVxYz5/ONyQ5FROKwfv16Ro0axahRo+jZsye9e/fe837Xrl31Hjtr1iwuvfTSmNcYP358s8T66quvctJJJzXLuZJBHdxRDuieR9vsLOZ/sZlJo3rHPkBEkqpr167MnTsXgOuuu468vDx+9KO96zlVVFSQk1P719y4ceMYN25czGu89dZbzRNsmlPNIkqb7CwO3D+P+Ss3JzsUEWmkadOmceGFF3LooYdy5ZVX8u677/KVr3yF0aNHM378eBYtWgTs+5f+ddddx7nnnsuECRMYOHAgt956657z5eXl7dl/woQJnHbaaQwdOpQpU6ZQPWv3M888w9ChQxk7diyXXnppg2oQDz74ICNGjGD48OFcddVVAFRWVjJt2jSGDx/OiBEj+P3vfw/ArbfeSnFxMQcddBBnnHFG0z+sBlDNoobiXhFeXbQ22WGIpKVfPDWP+V807x9bxb0iXPvNYQ06prS0lLfeeovs7Gw2b97M66+/Tk5ODi+99BI//elP+cc//vGlYxYuXMgrr7xCeXk5Q4YM4aKLLvrSfQjvv/8+8+bNo1evXhx++OG8+eabjBs3ju9///vMnDmTAQMGMHny5Ljj/OKLL7jqqquYPXs2nTt35rjjjuPxxx+nT58+rFixgo8//hiATZs2AXDzzTezdOlS2rVrt6espahmUUNRQYR1W3ayplw3GImkq9NPP53s7GwAysrKOP300xk+fDhXXHEF8+bNq/WYE088kXbt2tGtWzf2339/Vq9e/aV9DjnkEAoLC8nKymLUqFGUlJSwcOFCBg4cuOeehYYki/fee48JEybQvXt3cnJymDJlCjNnzmTgwIEsWbKESy65hOeee45IJALAQQcdxJQpU7j//vvrbF5LFNUsaigqyAdgwcpy9s/PTXI0IumloTWAROnQocOe1z/72c+YOHEijz32GCUlJUyYMKHWY9q1a7fndXZ2NhUVFY3apzl07tyZDz74gOef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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "AprPWAaz8qaO",
        "outputId": "6683b35d-18a4-4a8b-a3e9-02dac221b791"
      },
      "source": [
        "train_predictions = model_1.predict(X_train)\n",
        "test_predictions = model_1.predict(X_test)\n",
        "\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "print(\"Train Data: \"+str(accuracy_score(y_train.argmax(axis=-1), train_predictions.argmax(axis=-1))))\n",
        "print(\"Test Data: \"+str(accuracy_score(y_test.argmax(axis=-1), test_predictions.argmax(axis=-1))))"
      ],
      "execution_count": 46,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train Data: 0.8325\n",
            "Test Data: 0.26\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "mdKi9Nvl4WpR"
      },
      "source": [
        "## ResNet50"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "1YNcngoS4Xmv"
      },
      "source": [
        "#import inception with pre-trained weights. do not include fully #connected layers\n",
        "resnet50 = applications.ResNet50(weights='imagenet', include_top=False)\n",
        "\n",
        "# add a global spatial average pooling layer\n",
        "x = resnet50.output\n",
        "x = GlobalAveragePooling2D()(x)\n",
        "# add a fully-connected layer\n",
        "x = Dense(512, activation='relu')(x)\n",
        "# and a fully connected output/classification layer\n",
        "predictions = Dense(5, activation='softmax')(x)\n",
        "# create the full network so we can train on it\n",
        "resnet50 = Model(inputs=resnet50.input, outputs=predictions)"
      ],
      "execution_count": 47,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "RCbPCCCQ4cB0"
      },
      "source": [
        "# Compile\n",
        "resnet50.compile(loss='categorical_crossentropy',\n",
        "              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),\n",
        "              metrics=['accuracy'])"
      ],
      "execution_count": 48,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "aEGmho-17xL8",
        "outputId": "78645704-c566-4a32-85a0-83d8edf8f0d2"
      },
      "source": [
        "print(X_train.shape)\n",
        "print(y_train.shape)"
      ],
      "execution_count": 49,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "(400, 128, 128, 3)\n",
            "(400, 5)\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "kTsh9r5Z4eTC",
        "outputId": "cc5ceec3-726f-45e6-975e-59ac830cc020"
      },
      "source": [
        "history_resnet50 = resnet50.fit(X_train, y_train, epochs=10, shuffle = True, verbose = 1, validation_split=0.2)"
      ],
      "execution_count": 51,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/10\n",
            "10/10 [==============================] - 82s 8s/step - loss: 1.8133 - accuracy: 0.1937 - val_loss: 2.2659 - val_accuracy: 0.1875\n",
            "Epoch 2/10\n",
            "10/10 [==============================] - 86s 9s/step - loss: 1.6529 - accuracy: 0.2531 - val_loss: 2.1189 - val_accuracy: 0.2250\n",
            "Epoch 3/10\n",
            "10/10 [==============================] - 82s 8s/step - loss: 1.4517 - accuracy: 0.3656 - val_loss: 2.0192 - val_accuracy: 0.2375\n",
            "Epoch 4/10\n",
            "10/10 [==============================] - 82s 8s/step - loss: 1.2950 - accuracy: 0.5094 - val_loss: 1.9330 - val_accuracy: 0.2500\n",
            "Epoch 5/10\n",
            "10/10 [==============================] - 84s 8s/step - loss: 1.1316 - accuracy: 0.6313 - val_loss: 1.8582 - val_accuracy: 0.2750\n",
            "Epoch 6/10\n",
            "10/10 [==============================] - 85s 9s/step - loss: 1.0162 - accuracy: 0.7375 - val_loss: 1.7913 - val_accuracy: 0.3000\n",
            "Epoch 7/10\n",
            "10/10 [==============================] - 83s 8s/step - loss: 0.9223 - accuracy: 0.7781 - val_loss: 1.7326 - val_accuracy: 0.3125\n",
            "Epoch 8/10\n",
            "10/10 [==============================] - 82s 8s/step - loss: 0.8313 - accuracy: 0.8406 - val_loss: 1.6941 - val_accuracy: 0.2875\n",
            "Epoch 9/10\n",
            "10/10 [==============================] - 84s 8s/step - loss: 0.7758 - accuracy: 0.8625 - val_loss: 1.6603 - val_accuracy: 0.2875\n",
            "Epoch 10/10\n",
            "10/10 [==============================] - 84s 8s/step - loss: 0.6911 - accuracy: 0.8906 - val_loss: 1.6412 - val_accuracy: 0.2750\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "EjldvrfC-DHw",
        "outputId": "8f64325b-b116-4fe2-d9d9-60ea6696f435"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "# summarize history for accuracy\n",
        "plt.plot(history_resnet50.history['accuracy'], label='Training Accuracy')\n",
        "plt.plot(history_resnet50.history['val_accuracy'], label='Validation Accuracy')\n",
        "plt.title('Accuracy')\n",
        "plt.ylabel('Accuracy(Training and Validation)')\n",
        "plt.xlabel('epoch')\n",
        "plt.legend(loc='lower right')\n",
        "#plt.legend(['Training'],['Validation'], loc='upper left')\n",
        "plt.title('Training and Validation Accuracy')\n",
        "\n",
        "#plt.legend(['Pretrained'], loc='upper left')\n",
        "plt.show()\n",
        "# summarize history for loss\n",
        "plt.plot(history_resnet50.history['loss'], label='Training Loss')\n",
        "plt.plot(history_resnet50.history['val_loss'], label='Validation Loss')\n",
        "plt.title('Loss')\n",
        "plt.ylabel('Loss(Training and Validation)')\n",
        "plt.xlabel('epoch')\n",
        "#plt.legend(loc='upper right')\n",
        "plt.title('Training and Validation Loss')\n",
        "#plt.legend(loc='upper right')\n",
        "plt.legend(loc='upper right')\n",
        "plt.show()"
      ],
      "execution_count": 52,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4yLjIsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy+WH4yJAAAgAElEQVR4nOzdd3xUVfr48c9DAgmEgPQOidKRHroNRRcFYe2gKIiCXdF1d/3aFtuuP2VdO4qIiLpidxFRFBSxoBCKSK+BBAIEAiQQQso8vz/uTTIJyWQCmUzK83695pW5dZ4M4Tz3nnPuOaKqGGOMqbqqBTsAY4wxwWWJwBhjqjhLBMYYU8VZIjDGmCrOEoExxlRxlgiMMaaKs0Rg8hGRr0RkbGnvG0wiEiciQwJw3kUicrP7/joR+caffU/ic1qLyBERCTnZWI3xxRJBJeAWEjkvj4gc81q+riTnUtWLVfXt0t63PBKRB0RkcSHrG4pIhoic6e+5VPU9Vb2olOLKl7hUdaeq1lbV7NI4fyGfJyKyTUTWBeL8pvyzRFAJuIVEbVWtDewELvVa917OfiISGrwoy6V3gYEiEl1g/SjgD1VdE4SYguEcoDFwuoj0KcsPtr/J8sESQSUmIueJSIKI/F1E9gBviUg9EZkrIkkictB939LrGO/qjnEi8pOITHH33S4iF5/kvtEislhEUkVkgYi8IiLvFhG3PzE+ISI/u+f7RkQaem2/XkR2iMgBEXmoqO9HVROA74DrC2y6AZhVXBwFYh4nIj95LV8oIhtE5LCIvAyI17YzROQ7N779IvKeiJzmbnsHaA184d7R/U1EokREcwpNEWkuInNEJFlEtojIBK9zTxaRD0VklvvdrBWRmKK+A9dY4H/APPe99+/VRUS+dT9rr4g86K4PEZEHRWSr+znLRaRVwVjdfQv+nfwsIv8RkQPAZF/fh3tMKxH51P13OCAiL4tIDTemrl77NRaRNBFpVMzvawqwRFD5NQXqA22AiTj/5m+5y62BY8DLPo7vB2wEGgLPAG+KiJzEvv8FlgINgMmcWPh68yfGa4Ebca5kawD3A4hIZ2Cqe/7m7ucVWni73vaORUQ6AD3ceEv6XeWcoyHwKfAwznexFRjkvQvwLze+TkArnO8EVb2e/Hd1zxTyEbOBBPf4K4F/isj5XttHuPucBszxFbOI1HLP8Z77GiUiNdxtkcAC4Gv3s9oCC91D7wNGA5cAdYDxQJrPLyZPP2Ab0AR4ytf3IU67yFxgBxAFtABmq2qG+zuO8TrvaGChqib5GYfJoar2qkQvIA4Y4r4/D8gAwn3s3wM46LW8CLjZfT8O2OK1rRagQNOS7ItTiGYBtby2vwu86+fvVFiMD3st3w587b5/FKegyNkW4X4HQ4o4dy0gBRjoLj8F/O8kv6uf3Pc3AL967Sc4BffNRZz3z8DKwv4N3eUo97sMxSkks4FIr+3/Ama67ycDC7y2dQaO+fhuxwBJ7rnDgcPAZe620d5xFThuIzCykPW5sfr4nnYW8++d+30AA3LiK2S/fjhJU9zlWODqYP7/q6gvuyOo/JJUNT1nQURqicjrbtVJCrAYOE2K7pGyJ+eNquZc8dUu4b7NgWSvdQDxRQXsZ4x7vN6necXU3PvcqnoUOFDUZ7kxfQTc4N69XAfMKkEchSkYg3ovi0gTEZktIrvc876Lc+fgj5zvMtVr3Q6cK+UcBb+bcCm6Ln4s8KGqZrl/J5+QVz3UCudupjC+thUn3799Md9HK2CHqmYVPImq/obz+50nIh1x7ljmnGRMVZolgsqv4PCyfwE6AP1UtQ5OQyF41WEHQCJQ362GyNHKx/6nEmOi97ndz2xQzDFvA1cDFwKRwBenGEfBGIT8v+8/cf5durrnHVPgnL6GBN6N811Geq1rDewqJqYTuO0d5wNjRGSPOO1IVwKXuNVb8cDpRRweD5xRyPqj7k/vf+umBfYp+Pv5+j7igdY+Etnb7v7XAx97X/QY/1kiqHoiceq6D4lIfeAfgf5AVd2Bc9s+2W3kGwBcGqAYPwaGi8hZbl334xT/d/4jcAiYRl7986nE8SXQRUQudwuwu8lfGEYCR4DDItIC+GuB4/dSRAGsqvHAL8C/RCRcRLoBN+FcRZfU9cAmnGTXw321x6nGGo1TN99MRCaJSJiIRIpIP/fY6cATItJOHN1EpIE69fO7cJJLiIiMp/CE4c3X97EUJ7E+LSIR7u/s3d7yLnAZTjKYdRLfgcESQVX0PFAT2A/8itMQWBauw6nvPQA8CXwAHC9i35OOUVXXAnfgNPYmAgdxCjZfxyhOIdKG/IXJScWhqvuBq4CncX7fdsDPXrs8BvTCqY//Eqdh2du/gIdF5JCI3F/IR4zGqYvfDXwG/ENVF/gTWwFjgVdVdY/3C3gNGOtWP12Ik7T3AJuBwe6xzwEfAt/gtLG8ifNdAUzAKcwPAF1wEpcvRX4f6jw7cSlOtc9OnH/La7y2xwMrcO4ofiz5V2Agr5HFmDIlIh8AG1Q14HckpnITkRnAblV9ONixVFSWCEyZEOdBpWRgO3AR8DkwQFVXBjUwU6GJSBSwCuipqtuDG03FZVVDpqw0xelGeAR4EbjNkoA5FSLyBLAGeNaSwKmxOwJjjKni7I7AGGOquAo34FPDhg01Kioq2GEYY0yFsnz58v2qWug4TBUuEURFRREbGxvsMIwxpkIRkR1FbbOqIWOMqeIsERhjTBUX0EQgIkNFZKM4Y6Y/UMj2NiKyUERWu2OW+xou2BhjTAAELBG4IzS+AlyMMxTuaHeseG9TgFmq2g1nTJh/BSoeY4wxhQvkHUFfnPHpt2neJBIjC+zTGWeGKIDvC9lujDEmwAKZCFqQf9zxBPKPmQ7wO3C5+/4yIFJEihsy2BhjTCkKdmPx/cC5IrISOBdn+NrsgjuJyEQRiRWR2KQkm4XOGGNKUyATwS7yT8bRkgKTZ6jqblW9XFV7Ag+56w4VPJGqTlPVGFWNadTI5qU2xlQdaRlZ/LJlPy8s2Mza3YcD8hmBfKBsGdBORKJxEsAonAnHc7mzICWrqgf4P2BGAOMxxphyb19qOsvjDrIs7iDLdySzZncK2R5FBOrXrkGX5nVL/TMDlghUNUtE7gTmAyHADFVdKyKPA7GqOgdncvV/iYjizAd7R6DiMcaY8kZV2Zp0lOU7klkWd5DYuGTiDjhTe4eFVqNHq9O49dzTiYmqT6/W9ahbs3pA4qhwo4/GxMSoDTFhjKmIMrI8rNl9mNi4ZPeK/yDJR52ZUevVqk5MVH36RNUjJqo+ZzavS43Q0qu9F5HlqhpT2LYKN9aQMcZUFIePZbJi58Hcgv/3+EMcz/IAENWgFud3bEyfqHr0blOfMxpFICJBidMSgTHGlJJdh44RG5dMbNxBlsUls3FvKqoQUk3o0rwO1/Vr4xT8UfVoHBke7HBzWSIwxpiTkO1RNu1Nzb3aj41LZvfhdAAiaoTQq009Lj6zGX2i6tG91WlEhJXf4rb8RmaMMeVIemY2q+IP5Rb8K3YcJPV4FgCNI8PoE12fiW2c+v2OTSMJDQn2Y1r+s0RgjDEFHM/KZvPeI2zYk8r6xBRW7DzIml2Hycx2Ote0a1yb4d2b0yeqHn2i6tOyXs2g1e+XBksExpgqS1XZl3qc9YkprE9MZcOeFNYnprA16SjZHqfQDwutRtcWdbnprNPdht16nFarRpAjL12WCIwxVULBq3yn0E/N7b4J0LxuOB2b1eHCzk3o2LQOnZrVIapBrQpVzXMyLBEYYyoVVSUp9TjrirnK79A0kiGdGtOpmVPgd2waWemu9P1licAYU2HZVX7psERgjCn3vK/ycwv9xFS2Jh0hq4irfKfQr7pX+SVhicAYU+6oKj9vOcCijftYX8hVfrO64XRqVochnRvbVX4psERgjClXlm5PZsr8jSyNS7ar/DJiicAYUy6sTjjElG82sXhTEo0iw3h8ZBeu6dOKsNCQYIdW6VkiMMYE1aa9qfz7m43MX7uXerWq8+AlHbm+fxQ1a1gCKCt+JwIRiQDSVfWEqSSNMaak4vYf5fkFm/jf77uJqBHKvUPaM/6sKCLDAzPmvilakYlARKrhzCp2HdAHOA6Eich+4EvgdVXdUiZRGmMqjd2HjvHSd5v5MDaB6iHCxHNO59ZzzqBehNX7B4uvO4LvgQU4U0iucaeTRETqA4OB/ycin6nqu4EP0xhT0SWlHufVRVt477edoHB9/zbcPviMcjUcc1XlKxEMUdXMgitVNRn4BPhEROwezhjj0+G0TF5fvJW3fo4jI9vDlb1actcFbWlZr1awQzOuIhOBdxIQkRCgiff+qrqzsERhjDEAR45n8dZP25n24zaOHM/i0m7NmTSkHac3qh3s0EwBxTYWi8hdwD+AvYDHXa1AtwDGZYypoNIzs3n31x28umgryUczuLBzE/5yUXs6Nq0T7NBMEfzpNXQP0EFVD5T05CIyFHgBCAGmq+rTBba3Bt4GTnP3eUBV55X0c4wxwZeR5eHD2Hhe+m4ze1OOc3a7hvzlog70aHVasEMzxfAnEcQDh0t6Yrc66RXgQiABWCYic1R1ndduDwMfqupUEekMzAOiSvpZxpjgyfYon6/cxfMLNxGffIyYNvV4YVRP+p/eINihGT/5kwi2AYtE5EucLqQAqOpzxRzXF9iiqtsARGQ2MBLwTgQK5Nwv1gV2+xm3MSbIPB7l67V7eO7bTWzZd4QzW9Th8RvP5Lz2jSr0bF1VkT+JYKf7quG+/NUC524iRwLQr8A+k4Fv3HaICGBIYScSkYnARIDWrVuXIARjTGlTVRZtTGLKNxtZuzuFto1rM/W6Xgw9s6klgAqq2ESgqo8BiEhtd/lIKX7+aGCmqv5bRAYA74jImTnPLHjFMA2YBhATE6Ol+PnGmBJYsvUAU77ZyPIdB2ldvxbPXd2dkT1aEFLNEkBF5k+voTOBd4D67vJ+4AZVXVvMobuAVl7LLd113m4ChgKo6hIRCQcaAvv8it4YUyZWxR9iyvyN/LRlP03rhPPUZWdydUwrqtuwz5WCP1VD04D7VPV7ABE5D3gDGFjMccuAdiISjZMARgHXFthnJ3ABMFNEOgHhQJLf0RtjAmp9Ygr//mYTC9bvpUFEDR4e1okx/dsQXt0GhKtM/EkEETlJAEBVF7kD0PmkqlkicicwH6dr6AxVXSsijwOxqjoH+Avwhojci9NwPE5VrerHmCDblnSE/yzYzNzVu6kdFsr9F7XnxkHRRITZgMWVkV+9hkTkEZzqIYAxOD2JiuU+EzCvwLpHvd6vAwb5F6oxJpCOHM9i4fq9zF2dyHcb9hEWWo3bzzuDiWefQd1aNppMZeZPIhgPPAZ86i7/6K4zxlRwaRlZfLdhH1+6hf/xLA9N6oRx81nRTDjndBrWDgt2iKYM+NNr6CBwdxnEYowpA+mZ2SzauI8vVify3fp9HMvMplFkGKP6tGJ49+b0bl2PatYLqErxNR/B86o6SUS+wKm/z0dVRwQ0MmNMqUnPzGbxpiTmrk5k4fq9HM3IpkFEDa7o3YJhXZvTN7q+dQGtwnzdEeS0CUwpi0CMMaUrI8vDj5uT+HJ1It+u20vq8Szq1arOiB4tGN6tGf2i6xNq3T8NvoehXu6+7aGqL3hvE5F7gB8CGZgxpuQysz38vGU/c1cn8s3aPaSkZ1G3ZnUu7tqU4d2aM+CMBtb335zAn8bisTgjiHobV8g6Y0wQZGV7WLLtAF+uTuTrtXs4lJZJZHgoF3VuyvBuzRjUtiE1Qq3wN0Xz1UYwGucBsGgRmeO1KRJIDnRgxpiiZXuU37YdYO4fiXy9Zg/JRzOIqBHChZ2bMLxbc85u35CwUHvoy/jH1x3BL0AizpAP//ZanwqsDmRQxpgTZXuU2Lhk5q5O5Ks1e9h/5Di1aoRwQacmDO/WjHPbN7Infs1J8dVGsAPYAQwou3CMMd48HmXFzoPMXZ3IvD8S2Zd6nPDq1bigYxOGdWvG4A6NqVnDCn9zavwZdK4/8BLQCWcY6hDgqKravHPGBICqsir+UG7hn3g4nbDQagzu0Jhh3ZpxQafG1KphQz2Y0uPPX9PLOAPGfQTEADcA7QMZlDFVUfLRDF7/YStzVyey69AxaoRU45z2jXjg4o5c0KkJtW2cHxMgfv1lqeoWEQlR1WzgLRFZCfxfYEMzpurYceAoY2csJeHgMc5p34j7LmzPhV2aUCfcxvgxgedPIkgTkRrAKhF5BqcB2fqiGVNKfo8/xPiZy/Co8sEtA+jdpl6wQzJVjD8F+vU47QJ3AkdxJpu5IpBBGVNVLFy/l1HTfqVWWAif3DbQkoAJCn8Gndvhvj2GMwqpMaYUvPfbDh75fA1ntqjLm2P70CjSRvo0weHrgbI/KGSwuRyq2i0gERlTyakq//5mEy9/v4XBHRrxynW9rBeQCSpff33D3Z93uD+9J6axWcSMOQkZWR4e+HQ1n67Yxei+rXhi5Jk28JsJuuIeKENELlTVnl6b/i4iK4AHAh2cMZVJanomt727gp+27OcvF7bnzvPbImJDP5vg8+d+VERkkKr+7C4MxHoNGVMie1PSGffWMjbvTeXZK7txVUyrYIdkTC5/EsFNwAwRqQsIcBA/p6oUkaE4o5SGANNV9ekC2/8DDHYXawGNVfU0P2M3pkLYtDeVcTOWcvhYJjPG9eGc9o2CHZIx+fjTa2g50N1NBKjqYX9OLCIhwCvAhUACsExE5rgT1uec+16v/e8Cep5wImMqsF+3HWDirFjCqofw4a0D6NK8brBDMuYEvnoNjVHVd0XkvgLrAVDV54o5d19gi6puc4+bDYwE1hWx/2jgH37GbUy598Xvu/nLh7/TukEtZt7Yh5b1agU7JGMK5euOIML9GXmS524BxHstJwD9CttRRNoA0cB3RWyfCEwEaN269UmGY0zZUFWm/7idp+atp29Ufabd0JvTatUIdljGFMlXr6HX3Z9l8RDZKOBjdyyjwmKZBkwDiImJsa6rptzK9ihPzF3HzF/iGNa1Gf++urvNEWDKPV9VQy/6OlBV7y7m3LtwhqPI0dJdV5hR5D2vYEyFlJ6Zzb0frOKrNXu4+axoHrykE9WqWfdQU/75qhpa7mObP5YB7UQkGicBjMKZ+jIfEekI1AOWnOLnGRM0B49mMGFWLMt3HuSR4Z256azoYIdkjN98VQ29fSonVtUsEbkTmI/TfXSGqq4VkceBWFXNmQd5FDBbVa3Kx1RI8clpjH3LGUL65dG9GNatWbBDMqZE/JmhrBHwd6AzEJ6zXlXPL+5YVZ0HzCuw7tECy5P9jNWYcuePhMPcOHMZmdke3r2pH32j6wc7JGNKzJ8nhN8D1uP06nkMiMOp9jGmSvt+4z6umbaEsNBqfHLbAEsCpsLyJxE0UNU3gUxV/UFVxwPF3g0YU5l9sGwnN78dS3TDCD67fSBtG59sL2tjgs+fISYy3Z+JIjIM2A3YpY+pklSV5xds5oWFmzmnfSNeva6XzSVsKjxf3Uerq2om8KQ7vMRfgJeAOsC9RR1nTGWVme3hwU//4KPlCVzVuyX/vLwr1W0IaVMJ+LqU2SUic4D3gRRVXUPeAHHGVClHjmdx+3srWLwpiXsuaMekIe1sCGlTafi6nOmE0yj8MBAvIi+ISP+yCcuY8mNfSjrXvL6En7fs5/9d0ZV7L2xvScBUKkUmAlU9oKqvq+pgnAHktgH/EZGtIvJUmUVoTBBt2XeEy179he37jzJ9bAzX9LGxrkzl41cFp6ruBt4EpgKpwM2BDMqY8mBZXDJXTP2F41kePpg4gMEdGgc7JGMCwmciEJFwEblKRD4FtuB0G30AaF4WwRkTLF/9kch103+jQUQNPrt9IF1b2jwCpvLy1Wvov8AQ4Aech8quVdX0sgrMmGCZ8dN2nvhyHb1a12P6DTHUi7AhpE3l5qvX0NfALaqaWlbBGBNMHo/yz3nrmf7Tdv7UpQkvjOppQ0ibKsHXoHOzyjIQY4IpPTObv3z0O1+uTmTcwCgeGd6ZEBtC2lQR9kikqfJS0zO56e1Ylm5P5qFLOnHz2dHWPdRUKZYITJV25HgWY2csZXXCYV4c3ZMR3a0fhKl6fDUWX+7rQFX9tPTDMabsHDmexbgZS/k94TCvXNuLoWc2DXZIxgSFrzuCS92fjYGB5E0sPxj4BbBEYCqstIwsxr+1jJXxh3hpdE9LAqZK89VYfCOAiHwDdFbVRHe5GTCzTKIzJgCOZWQzfuYyYnck88KonlzS1WYUM1WbP08Wt8pJAq69gD1nbyqkYxnZ3PT2MpZuT+Y/1/TgUmsTMMavxuKFIjIfZxRSgGuABYELyZjASM/MZsKsWJZsO8BzV3dnZI8WwQ7JmHKh2DsCVb0TeB3o7r6mqepd/pxcRIaKyEYR2SIiDxSxz9Uisk5E1rpPMxtT6tIzs5n4znJ+3rqfZ6/szmU9WwY7JGPKDb+6j7o9hErUOCwiIcArwIVAArBMROao6jqvfdoB/wcMUtWDImKjeplSdzwrm9veXc7iTUk8c0U3ruxtScAYb8XeEYjI5SKyWUQOi0iKiKSKSIof5+4LbFHVbaqaAcwGRhbYZwLwiqoeBFDVfSX9BYzxJSPLw+3vruD7jUn86/KuXN2nVbBDMqbc8aex+BlghKrWVdU6qhqpqnX8OK4FEO+1nOCu89YeaC8iP4vIryIytLATichEEYkVkdikpCQ/PtoYJwnc8d8VLNywjyf/fCaj+1ofB2MK408i2Kuq6wP0+aFAO+A8YDTwhoicVnAnVZ2mqjGqGtOoUaMAhWIqk8xsD3e9v4Jv1+3l8ZFdGNO/TbBDMqbc8qeNIFZEPgA+B47nrPTjyeJdgPd9eEt3nbcE4DdVzQS2i8gmnMSwzI+4jClUVraHe2avZP7avfzj0s7cMCAq2CEZU675c0dQB0gDLsJ52vhSYLgfxy0D2olItIjUAEYBcwrs8znO3QAi0hCnqmibX5EbU4isbA+TPljFvD/28PCwTtw4KDrYIRlT7hV7R5DzhHFJqWqWiNwJzAdCgBmqulZEHgdiVXWOu+0iEVkHZAN/VdUDJ/N5xmR7lL989DtzVyfy4CUdufns04MdkjEVgqiq7x1EwoGbgC5AeM56VR0f2NAKFxMTo7GxscH4aFOOZXuUv370O5+u3MXfhnbg9vPaBjskY8oVEVmuqjGFbfOnaugdoCnwJ5xpK1viTGBvTLng8Sh//2Q1n67cxf0XtbckYEwJ+ZMI2qrqI8BRVX0bGAb0C2xYxvjH41Ee+HQ1Hy9P4N4h7bnz/HbBDsmYCsefRJDp/jwkImcCdXGGpjYmqDwe5aHP/+DD2ATuPr8t9wyxJGDMyfCn++g0EakHPIzT66c28EhAozKmGKrKo3PW8P7SeO4YfAb3Xtg+2CEZU2H502touvt2MWDdMEzQqSr/mLOWd3/dya3nnsH9F3WwOYaNOQX+VA0ZU26oKo/PXcesJTuYcHY0fx9qScCYU2WJwFQYqspTX67nrZ/jGD8omgcv6WRJwJhSYInAVAiqytNfbWD6T9sZNzCKR4ZbEjCmtBTbRiAilxey+jDwhw0bbcqCqvLM/I28vngb1/dvwz8u7WxJwJhS5E+voZuAAcD37vJ5wHIgWkQeV9V3AhSbMagqz327iamLtnJtv9Y8NqKLJQFjSpk/iSAU6KSqewFEpAkwC+ehssU4Tx4bExDPL9jMS99tYVSfVjw58kyqVbMkYExp86eNoFVOEnDtc9clk/ewmTGl7sWFm3lh4Wau6t2Sf17W1ZKAMQHizx3BIhGZC3zkLl/hrosADgUsMlOlvfL9Fp77dhOX92rB01d0syRgTAD5kwjuwCn8B7nLs4BP1Bm2dHCgAjNV12s/bOXZ+Rv5c4/mPHtld0IsCRgTUP48WazAx+7LmIB6Y/E2nv5qA5d2b86UqywJGFMWim0jEJHLRWSziBwWkRQRSRWRlLIIzlQtb/60nafmrWdYt2b85+ruhIbYYy7GlAV/qoaeAS4N4AT2xjDz5+08MXcdF5/ZlOev6WFJwJgy5M//tr2WBEwgvbMkjslfrOOizk14cXRPqlsSMKZM+XNHECsiH+BMNH88Z6WqfhqwqEyV8d5vO3jkf2sZ0qkJL1/by5KAMUHgz/+6OkAacBFwqfsa7s/JRWSoiGwUkS0i8kAh28eJSJKIrHJfN5ckeFOxfbtuLw9/vobzOzbmlet6UiPUkoAxweBPr6EbT+bEIhICvAJcCCQAy0RkjqquK7DrB6p658l8hqm41u1O4Z7ZK+nWoi6vXteLsNCQYIdkTJVVZCIQkb+p6jMi8hKgBber6t3FnLsvsEVVt7nnmw2MBAomAlPF7EtN5+a3l1EnvDrTboghvLolAWOCydcdQU4DcexJnrsFEO+1nEDhk95fISLnAJuAe1U1vuAOIjIRmAjQunXrkwzHlAfpmdnc8s5yktMy+PjWgTSpEx7skIyp8opMBKr6hfvz7QB+/hfA+6p6XERuAd4Gzi8klmnANICYmJgT7k5MxaCqPPDJalbuPMTU63pxZou6wQ7JGIN/8xG0B+4Horz3V9UTCuwCdgGtvJZbuutyqeoBr8XpOM8smErqle+38Pmq3dx/UXsu7tos2OEYY1z+dB/9CHgNp6DOLsG5lwHtRCQaJwGMAq713kFEmqlqors4grzqKFPJfPVHIlO+2cSfezTnjsFtgx2OMcaLP4kgS1WnlvTEqpolIncC84EQYIaqrhWRx4FYVZ0D3C0iI4AsIBkYV9LPMeXfml2HuffDVfRsfRpPX9HNJpYxppwRZ0w5HzuITMaZg+Az8j9QlhzQyIoQExOjsbEn235tytrelHRGvvwz1QT+d+dZNIoMC3ZIxlRJIrJcVWMK2+bPHcFY9+dfvdYpcPqpBmYqt2MZ2UyYFUtKeiaf3DbQkoAx5ZQ/D5RFl0UgpnLxeJT7P/6dP3YdZtr1MXRqVifYIRljiuDrgbLzVfU7Ebm8sO021pDx5YWFm/lydSL/d3FHLuzcJNjhGGN88HVHcC7wHc7YQgUpYInAFGrO77t5YeFmruzdkonnWA2iMeWdrwfK/uH+PKmxhkzVtCr+EH/96Hf6RtXnqXn4s3kAACAASURBVMvOtB5CxlQA/jQWIyLDgC5A7ngAqvp4oIIyFdPuQ8eYMCuWxnXCmDrGBpIzpqLw58ni14BaOBPVTweuBJYGOC5TwaRlZHHz27Ecy8jmvZv70aC29RAypqLwZwD4gap6A3BQVR8DBgDtAxuWqUg8HuXeD1axYU8KL13bk/ZNIoMdkjGmBPxJBOnuzzQRaQ5kAjZQjMn17283Mn/tXh4a1pnBHRoHOxxjTAn500bwhYicBjwLrMDpMfRGQKMyFcZnKxN45futjO7bmvGDooIdjjHmJPhMBCJSDVioqoeAT0RkLhCuqofLJDpTri3fkczfP/6D/qfX5/GRXayHkDEVlM+qIVX14Ew3mbN83JKAAUg4mMbEWctpflo4U6/rbZPOG1OB+fO/d6GIXCF2uWdcR447PYQysj1MH9uHehE1gh2SMeYUFJkIROSf7ttbcOYkOC4iKSKSKiIpZRKdKXeyPco9769k874jvHpdL9o2rh3skIwxp8jXHcFQAFWNVNVqqlpDVeu4yzaCWBX1/77ewMIN+5h8aWfObtco2OEYY0qBr8biEBGpBxRaJRSs+QhM8Hy4LJ5pi7dxw4A2XD8gKtjhGGNKia9E0BFYTuGJwOYjqGJ+23aAhz7/g7PaNuTR4Z2DHY4xphT5SgTrVLVnmUViyq2dB9K49d3ltKpfi1eu60Wo9RAyplKx/9HGp5T0TG56exkehRlj+1C3ZvVgh2SMKWW+EsELp3pyERkqIhtFZIuIPOBjvytEREWk0Pk0TXBkZXu4678r2b7/KFPH9CKqYUSwQzLGBICvRHCWiHQtbIOIRIjIeBG5rqiDRSQE52G0i4HOwGgROaFyWUQigXuA30oUuQm4p+at54dNSTzx5zMZeEbDYIdjjAkQX20ELwOPuMlgDZCEMx9BO6AOMAN4z8fxfYEtqroNQERmAyOBdQX2ewL4f8BfT+YXMIHx3m87eOvnOMYPimZ039bBDscYE0C+ZihbBVwtIrWBGJwRR48B61V1ox/nbgHEey0nAP28dxCRXkArVf1SRIpMBCIyEZgI0Lq1FUqB9suW/fzjf2sZ3KERDw3rFOxwjDEB5s/oo4OBL91xh0qNO6Ddc8C44vZV1WnANICYmBgtzThMftuSjnDbeyuIbhjBi6N7ElLNRhYxprLzp9fQNcBmEXlGRDqW4Ny7gFZeyy3ddTkigTOBRSISB/QH5liDcfAcTsvk5rdjCakmvDm2D5Hh1kPImKqg2ESgqmOAnsBWYKaILBGRiW4jry/LgHYiEi0iNYBRwByv8x5W1YaqGqWqUcCvwAhVjT3ZX8acvMxsD7f/dznxB9N4bUxvWjeoFeyQjDFlxK/nCFQ1BfgYmI3TVnAZsEJE7vJxTBZwJzAfWA98qKprReRxERlxypGbUvX4F+v4ecsB/nlZV/pG1w92OMaYMuTP5PUjgBuBtsAsoK+q7hORWjg9gF4q6lhVnQfMK7Du0SL2Pc//sE1pmrUkjnd+3cEt557OVTGtit3fGFO5+NNYfAXwH1Vd7L1SVdNE5KbAhGXKyuJNSTz2xTqGdGrC3/5UkiYgY0xl4U8imAwk5iyISE2giarGqerCQAVmAm/LvlTu+O8K2jWuzfOjelgPIWOqKH/aCD4CvLuOZrvrTAV28GgGN70dS1hoNaaPjaF2mD/XBMaYysifRBCqqhk5C+57m5uwAktNz2TCrFgSD6fz+vUxtKxnPYSMqcr8SQRJ3r18RGQksD9wIZlAOnDkOKPf+JVV8Yd4/poe9G5TL9ghGWOCzJ/6gFuB90TkZZxJauKBGwIalQmIxMPHGDP9NxIOHuONG2IY3LFxsEMyxpQDxSYCVd0K9HfHHEJVjwQ8KlPqtu8/ypjpv5FyLJN3bupnzwoYY3L51UIoIsOALkC4iNOzRFUfD2BcphStT0zh+jeX4lHl/Yn9ObNF3WCHZIwpR4ptIxCR13DGG7oLp2roKqBNgOMypWT5joNc8/oSqocIH94ywJKAMeYE/jQWD1TVG4CDqvoYMABoH9iwTGn4cXMSY6b/Rv2IGnx06wDaNq4d7JCMMeWQP4kg3f2ZJiLNgUyc8YZMOfbVH4mMn7mMqIYRfHTrQOsiaowpkj9tBF+IyGnAs8AKQIE3AhqVOSUfxsbzwCer6dHqNN4a15e6tWw4aWNM0XwmAnfymIWqegj4RETmAuGqerhMojMl9uZP23li7jrObteQ16/vTa0a9sSwMcY3n1VD7qxkr3gtH7ckUD6pKs99u4kn5q7j4jObMn1sjCUBY4xf/GkjWCgiV0hOv1FT7ng8ymNfrOPFhZu5qndLXhrdk7DQkGCHZYypIPy5ZLwFuA/IEpF0nC6kqqp1AhqZ8UtWtoe/fbKaT1fs4qazonnokk5Us1FEjTEl4M+TxcVNSWmCJD0zm7vfX8k36/Zy34Xtuev8ttiNmzGmpPyZoeycwtYXnKjGlK2jx7OY+E4sP285wORLOzNuUHSwQzLGVFD+VA391et9ONAXWA6cH5CITLEOpWUw7q1l/LHrMP++qjtX9G4Z7JBMRaEKacmQuhtSEiHV65WS6KxP3QOebGjWHVr0gha9oXkvqGOPD1VW/lQNXeq9LCKtgOf9ObmIDAVeAEKA6ar6dIHttwJ34Ex2cwSYqKrr/Au9atqXks71by5l+/6jvHpdL/7UpWmwQzLlRUZa4YV6ym6vAn8PZGeceGxEI4hsCpHNnYJfPbB7Jfz0PGi2s09kMychtOjpJoeeUNOGMa8MTqZ/YQLQqbidRCQEp+vphe4xy0RkToGC/r+q+pq7/wjgOWDoScRUJcQnpzHmzd9ISj3OWzf2YVDbhsEOyZSF7Cw4us+rgE8s/P3xQnp2V49wruQjm0Gr/u775k6hX6e5s752EwgtYq6pzGOw5w/YtRx2rYDdK2Djl3nb65/uJofezt1D025Qw55ir2j8aSN4CedpYnC6m/bAecK4OH2BLaq6zT3PbGAkkJsIVDXFa/8Ir88xBWzem8qYN38jPdPDezf3o2druxKrVI7scwravWvcK/g9eVf0R/Y6V+jeJMS9gm8GDdpC9Dl5V/Q5BX9kMwg/xc591WtCq77OK8exQ87dwu4VTsw7l8Caj/PiatzJSQrNezk/G3eGEHu6vTzz544g1ut9FvC+qv7sx3EtcCaxyZEA9Cu4k4jcgdM9tQZFtDuIyERgIkDr1q39+OjKZXXCIcbOWEpoSDU+uKU/HZtaz90KLf0w7F7lFqTLYddKSEnI216zXt5Ve5MueYV6neZ5hX1EQ6gWpGdFap4GZwx2XjlS9+TdMexaAevmwIpZzrbQcGjaNa+toUUvqH8GVPPnMSZTFkTV90W4iEQA6apORaFb5ROmqmnFHHclMFRVb3aXrwf6qeqdRex/LfAnVR3r67wxMTEaGxvra5dKZcnWA0yYFUu9iOq8e1M/2jSICHZIpiQy052qldxCfwUc2Jy3vV5U/gKyaTcIqwSjxKrCwe1ucljp/O6Jv0OmW2yE1YXmPfLfOdRpAdb9OWBEZLmqxhS2zZ87goXAEJzGXICawDfAwGKO2wW08lpu6a4rymxgqh/xVBkL1u3l9v+uoE39WrxzUz+a1g0PdkjGl+wsSNqQv9Dftw48Wc722k2cQr/bNU6Da/NeUKuSzhQn4rQf1D8dul7prMvOgv0bve4clsMvL+V9PxGN89oacpJDZf1+yhl/EkG49/SUqnpERPxpDVoGtBORaJwEMAq41nsHEWmnqjmXR8OAzRgAPl+5i7989Dtdmtdh5o19qR9RRGOeCQ5VSN6Wd7W7awXsWZ3/irdFTxh4d17BVqd51b7iDQl1qrqadIFe1zvrMtOddhHv5LDpa3KbC2vUxhnMIMjC67oN7M0Kr6qLbHrq7TFB5E8iOCoivVR1BYCI9AaOFXeQqmaJyJ3AfJzuozNUda2IPA7Equoc4E4RGYIzx8FBwGe1UFXxzpI4Hp2zln7R9Xnjhhgiw62hLehSEvNf6e9eCemHnG2h4U6f+15j8wr9+qcHvA48MzOThIQE0tPTi9+5XIuAOmc7r444DePZGZCVkdd1NdjU4zxbodnuTw+kAWnHYe92YDtINaexvJr7Kuy9hAT8YiA8PJyWLVtSvbr/5YY/bQR9cKptduOk5qbANaq6/BRiPWmVuY1AVXl10Vaenb+RIZ0a8/K1vQivboPHlbljB72u9N3eMamJzjYJgSad86oumvdyeskEoVfM9u3biYyMpEGDBja0SFnzZIMnE7Ldl/d77+XCOkJWC4Vq1Z2/mZyX93K1Gm7SKPm/qapy4MABUlNTiY7OP9rAKbURqOoyEekIdHBXbVTVzBJHaHxSVZ7+agOvL97Gn3s059mrulM9xHpVBFxGmlOls8u92t+9wqnyydGgLUSdnVfoN+1abvrJp6enExUVZUkgGHKu9EN9tNupOu0fvhJGZlpeG0k+cmKCCKkOYXWcLr1FEBEaNGhAUlJSiX4df54juAN4T1XXuMv1RGS0qr5aok8yRcr2KA9//gfvL43n+v5teGxEFxtBNBCyM53GW+/qnX3r86of6rRwnpbtOcYp9Jv3dLpKlmOWBMoxkbwC3NcNo3rcJJHlVIkVTBiZx+B4irNf3RCficD52JL/TfjTRjBBVb0npzkoIhMASwSlICPLw70fruLL1YncMfgM7r+og/3nLg0eDyRvzf9E7J4/IMutT69ZzynsO1ycV80TacN1mCCQahAaBoThPFdbBE/g2kv8SQQhIiLqNia4zxFYF5ZScCwjm9veW86ijUn838UdueXcM4IdUsWkCim78hf6u1c5V1EA1WtBsx7Q52bnKr9Fb6f/viXcU3LgwAEuuOACAPbs2UNISAiNGjUCYOnSpdSoUXQxERsby6xZs3jxxRd9fsbAgQP55ZdfSi3mSZMm8dFHHxEfH0+1ivZAWwAfIPQnEXwNfCAir7vLt7jrzClISc/kppnLiN1xkH9d3pXRfaveE9Mn7eiBvCdYc34e3edsq1bd6Z7Y9cq8MXAadQjeU7iVWIMGDVi1ahUAkydPpnbt2tx///2527OysggNLbyIiYmJISam0HbLfEozCXg8Hj777DNatWrFDz/8wODBg4s/6CT4+r3LK3+i/TvO8A63ucvfAm8ELKIqYP+R44ydsZRNe1N5aXRPhndrHuyQyq/jRyBxVf5C/9AOd6NAw/bQ9oK8Qr9JF6he9R68e+yLtazbnVL8jiXQuXkd/nFplxIdM27cOMLDw1m5ciWDBg1i1KhR3HPPPaSnp1OzZk3eeustOnTowKJFi5gyZQpz585l8uTJ7Ny5k23btrFz504mTZrE3XffDUDt2rU5cuQIixYtYvLkyTRs2JA1a9bQu3dv3n33XUSEefPmcd999xEREcGgQYPYtm0bc+fOPSG2RYsW0aVLF6655href//93ESwd+9ebr31VrZtczoJTJ06lYEDBzJr1iymTJmCiNCtWzfeeecdxo0bx/Dhw7nyyitPiO+RRx6hXr16bNiwgU2bNvHnP/+Z+Ph40tPTueeee5g4cSIAX3/9NQ8++CDZ2dk0bNiQb7/9lg4dOvDLL7/QqFEjPB4P7du3Z8mSJbl3WIHmT68hD/Ca+0JEzgZewhk+2pRQfHIaY2csZffhY7xxQwzndWgc7JDKj6zjXg8XrXR+7t+YN+Ba3dbOQ1ox451Cv1n3Cv0QT2WVkJDAL7/8QkhICCkpKfz444+EhoayYMECHnzwQT755JMTjtmwYQPff/89qampdOjQgdtuu+2EfvArV65k7dq1NG/enEGDBvHzzz8TExPDLbfcwuLFi4mOjmb06NFFxvX+++8zevRoRo4cyYMPPkhmZibVq1fn7rvv5txzz+Wzzz4jOzubI0eOsHbtWp588kl++eUXGjZsSHJycrG/94oVK1izZk1ut80ZM2ZQv359jh07Rp8+fbjiiivweDxMmDAhN97k5GSqVavGmDFjeO+995g0aRILFiyge/fuZZYEwM9hqEWkJzAauBrYDnwayKAqo92HjvHaD1uZvSyesJBqvHNTP/pEVcHH51Wdfvo5Y+Sn7HKHOXZH3swZK79WQ6cBt8uf83rw1C67/xgVTUmv3APpqquuIiTEqYo7fPgwY8eOZfPmzYgImZmF9zwfNmwYYWFhhIWF0bhxY/bu3UvLlvknXOrbt2/uuh49ehAXF0ft2rU5/fTTcwvf0aNHM23atBPOn5GRwbx583juueeIjIykX79+zJ8/n+HDh/Pdd98xa5YzQF5ISAh169Zl1qxZXHXVVTRs6Az1Xr9+8f9X+/btm6/v/osvvshnn30GQHx8PJs3byYpKYlzzjknd7+c844fP56RI0cyadIkZsyYwY033ljs55WmIhOBiLTHKfxHA/uBD3AeQAtMxVollXAwjamLtvJRbAIeVa7s3ZI7BrelVf3y0Re9VHlPjJI7IcqeArNh7YHs4/mPqxHpDEDW/7a8Hjx1W1ljbgUVEZHX8+WRRx5h8ODBfPbZZ8TFxXHeeecVekxYWFju+5CQELKyTuxb788+RZk/fz6HDh2ia9euAKSlpVGzZk2GDx/u9zkAQkND8XicO1SPx0NGRt4kP96/96JFi1iwYAFLliyhVq1anHfeeT6fAG/VqhVNmjThu+++Y+nSpbz33nsliutU+boj2AD8CAxX1S0AInJvmURVCcQnp/Hqoi18vNwZXvjqmFbcdt4ZtKxXAROAJ9sZLz93EpScAj4xf2GfXtjEKLXyxmVp1c9rQhSvsfPrtLQhiSupw4cP06JFCwBmzpxZ6ufv0KED27ZtIy4ujqioKD744INC93v//feZPn16btXR0aNHiY6OJi0tjQsuuICpU6cyadKk3Kqh888/n8suu4z77ruPBg0akJycTP369YmKimL58uVcffXVzJkzp8g7nMOHD1OvXj1q1arFhg0b+PXXXwHo378/t99+O9u3b8+tGsq5K7j55psZM2YM119/fe4dVVnxlQguxxko7nsR+RpnmAm7RCvGjgNHeeX7LXy6YhfVRBjdtzW3nnsGzU/z/RBIUKg6hfcJV+2J+ac4LGpilNpNnIK8wRkQdVb+AbkimznLYXXsyr4K+9vf/sbYsWN58sknGTZsWKmfv2bNmrz66qsMHTqUiIgI+vTpc8I+aWlpfP3117z22mu56yIiIjjrrLP44osveOGFF5g4cSJvvvkmISEhTJ06lQEDBvDQQw9x7rnnEhISQs+ePZk5cyYTJkxg5MiRdO/ePfczCzN06FBee+01OnXqRIcOHejfvz8AjRo1Ytq0aVx++eV4PB4aN27Mt99+C8CIESO48cYby7xaCPyfj2AkThXR+cAs4DNV/Sbw4Z2ovI41tC3pCC9/v4X/rdpNaLW8BBC0oaOzjuddtZ8wh61XgZ9ZyLQS4aedeNXuPcpinebOHLfWJTOo1q9fT6dOxc4aW+kdOXKE2rVro6rccccdtGvXjnvvrXiVF7Gxsdx77738+OOPp3yuwv42TnWsoaPAf4H/ikg94CqcLqVBSQTlzZZ9qbz83Rbm/L6bGqHVuHFgFBPPOZ3GdQKUADweSDtQYGJy7yt6933agROPDQnLK8ibdYf2QwsMq9sMajctN2PpGOOPN954g7fffpuMjAx69uzJLbfcEuyQSuzpp59m6tSpZd42kKPYO4LyprzcEWzam8pL321h7urdhIeGcMOANtx89uk0igwr/uCiHD9SxMTkXnXyqXuccUjyEecK/YSJyQvMYVuznlXTVCJ2R2CKUup3BCa/9YkpvPTdZub9sYeIGiHceu4Z3HxWNA1q+5EAPB5IWOZM2Zfvit4t4I8X8kBQjci8grzNoMInxqjdxCYHN8acNEsEflq7+zAvLtzM/LV7iQwL5a7z2zJ+UDT1ips5LDMdti+GDXNh41deQyGEugV6U2jUEc44/8Qr+MimEBYZ+F/OGFOlWSIoxuqEQ7y4cAsL1u8lMjyUey5ox/hB0dSt5eMK/NhB2PytU/hvXgCZR50p99pdCB2HOz1sIhpbl0ljTLlgiaAIq+IP8cKCTXy/MYm6Natz34XtGTswiro1i0gAhxNgwzzY+CXE/eSMLV67CXS7GjoOg+hz3KFmjTGmfLFL0gKW7zjIDTOW8udXfmZV/CH++qcO/PT3wdx9Qbv8SUAV9q6DH56F18+F/3SBr/4Kh3fBgDvhpgVw3wa49HnnTsCSgKlkBg8ezPz58/Ote/7557ntttuKOALOO+88cjp7XHLJJRw6dOiEfSZPnsyUKVN8fvbnn3/OunXrcpcfffRRFixYUJLwfZo0aRItWrTIfYq4sgvoHYGIDAVewJm8frqqPl1g+33AzUAWkASMV9UdJ5yoDCzdnsyLCzfz05b91I+owQMXd2RM/zbUDvP6ijzZEP8bbPjSeR3c7qxv2QeGTIYOw6BR+2CEb0yZGz16NLNnz+ZPf/pT7rrZs2fzzDPP+HX8vHnzTvqzP//8c4YPH07nzp0BePzxx0/6XAVVxeGqAxaFO4HNK8CFQAKwTETmqOo6r91WAjGqmiYitwHPANcEKqbCLNl6gBcXbmbJtgM0rF2Dhy7pxHX9W1OrhvvVZB6Drd87VT4bv4a0/RBSA6LPhUF3Q4dLbGYrE3xfPeAM3leamnaFi58ucvOVV17Jww8/TEZGBjVq1CAuLo7du3dz9tlnc9ttt7Fs2TKOHTvGlVdeyWOPPXbC8VFRUcTGxtKwYUOeeuop3n77bRo3bkyrVq3o3bs34DwjMG3aNDIyMmjbti3vvPMOq1atYs6cOfzwww88+eSTfPLJJzzxxBO5w0MvXLiQ+++/n6ysLPr06cPUqVMJCwsjKiqKsWPH8sUXX5CZmclHH31Ex44dT4irKg5XHch01BfYoqrbAERkNs4TyrmJQFW/99r/V2BMAOPJpar8svUALyzczNLtyTSKDOOR4Z25tm9ratYIgbRkWDffaezd+p3z9G1YHWh3kVPf33aIDX9sqrz69evTt29fvvrqK0aOHMns2bO5+uqrERGeeuop6tevT3Z2NhdccAGrV6+mW7duhZ5n+fLlzJ49m1WrVpGVlUWvXr1yE8Hll1/OhAkTAHj44Yd58803ueuuuxgxYkS+gjZHeno648aNY+HChbRv354bbrghdxwhgIYNG7JixQpeffVVpkyZwvTp00+IpyoOVx3IRNACiPdaTgD6+dj/JuCrwjaIyEScyXFo3frkZ/JSVX7cvJ8XF24mdsdBmtQJY/KlnRnVtzXhRxJgxTSnymfHL86E5pHNoMe1TuHf5iwItRk6TTnl48o9kHKqh3ISwZtvvgnAhx9+yLRp08jKyiIxMZF169YVmQh+/PFHLrvsMmrVcp5oHzFiRO62NWvW8PDDD3Po0CGOHDmSrxqqMBs3biQ6Opr27Z0q2rFjx/LKK6/kJoLLL78cgN69e/PppyeOpl9Vh6suFxVUIjIGiAHOLWy7qk4DpoHzZPHJfMav2w7w9FcbWBV/iOZ1w3liZBeubnWYsC0fwJtz826rG3WEsyY5hX+zntbF0xgfRo4cyb333suKFStIS0ujd+/ebN++nSlTprBs2TLq1avHuHHjfA7B7Mu4ceP4/PPP6d69OzNnzmTRokWnFG/OUNZFDWNdVYerDmQptwto5bXc0l2Xj4gMAR4CRqjq8YLbS8vO5DSSU9KYfk46i7vP5/pfLyVs+jmw6GmoHgEXPgF3rYA7foMLHnVmwLIkYIxPtWvXZvDgwYwfPz53iOeUlBQiIiKoW7cue/fu5auvCr3Rz3XOOefw+eefc+zYMVJTU/niiy9yt6WmptKsWTMyMzPzFXqRkZGkpqaecK4OHToQFxfHli1bAHjnnXc499xCry8LlTNcdVxcHHFxcWzfvp1vv/0233DVANnZ2Rw+fJjzzz+fjz76iAMHnLG9cqqGcoarBk56uOrFixezffv2fOeFvOGqvScAOlWBLOmWAe1EJFpEauAMaT3Hewd35rPXcZLAvgDGwhUs5IdqExiydDyhy2dCk84w4iW4fxPcNN9p+G1wRiBDMKZSGj16NL///ntuIujevTs9e/akY8eOXHvttQwaNMjn8b169eKaa66he/fuXHzxxfmGkn7iiSfo168fgwYNytewO2rUKJ599ll69uzJ1q1bc9eHh4fz1ltvcdVVV9G1a1eqVavGrbfe6tfvkTNctfdw2QWHq/7+++/p2rUrvXv3Zt26dXTp0iV3uOru3btz3333ATBhwgR++OEHunfvzpIlS3wOV52VlUWnTp144IEHCh2uunv37lxzTV4fmhEjRnDkyJFSHa46oIPOicglwPM43UdnqOpTIvI4EKuqc0RkAdAVSHQP2amqI4o4HXAKg85tXgB/fOhU+ZxxAYTVLvk5jClHbNC5qsmf4arL1aBzqjoPmFdg3aNe74cE8vPzaTfEeRljTAUVqOGqrRLcGGMqiAceeIAdO3Zw1llnlep5LREYU4FVtPlETOCdzN+EJQJjKqjw8HAOHDhgycDkUlUOHDhAeHjJZkgsF88RGGNKrmXLliQkJJCUlBTsUEw5Eh4eTsuWLUt0jCUCYyqo6tWr53tC1ZiTZVVDxhhTxVkiMMaYKs4SgTHGVHEBfbI4EEQkCTjZyWsaAvtLMZyKzr6P/Oz7yGPfRX6V4ftoo6qFjlld4RLBqRCR2KIesa6K7PvIz76PPPZd5FfZvw+rGjLGmCrOEoExxlRxVS0RTAt2AOWMfR/52feRx76L/Cr191Gl2giMMcacqKrdERhjjCnAEoExxlRxVSYRiMhQEdkoIltE5IFgxxMsItJKRL4XkXUislZE7gl2TOWBiISIyEoRmRvsWIJNRE4TkY9FZIOIrBeRAcGOKVhE5F73/8kaEXlfREo2rGcFUSUSgYiEAK8AFwOdgdEi0jm4UQVNFvAXVe0M9AfuqMLfhbd7uzW7RAAAA8xJREFUgPXBDqKceAH4WlU7At2pot+LiLQA7gZiVPVMnCl3RwU3qsCoEokA6AtsUdVtqpoBzAZGBjmmoFDVRFVd4b5PxflP3iK4UQWXiLQEhgHTgx1LsIlIXeAc4E0AVc1Q1UPBjSqoQoGaIhIK1AJ2BzmegKgqiaAFEO+1nEAVL/wARCQK6An8FtxIgu554G+AJ9iBlAPRQBLwlltVNl1E/n979xOiVRWHcfz7iCnNTGBBLVJJ+4NEYFoQ4SCI4zKihRGUg7Q2wV0YRtA6ahXqJphwFpKN4EJIVBhwUVo2JdlOxUaMaSGGgWHytLgneBsNJnHmDHOez+p9z3vn8rswd557z5n3d/trF1WD7SvAR8Bl4Cpw3faxulXNjlaCIKaRNAB8Ceyy/XvtemqR9AowZfu72rXME4uBF4C9ttcDfwBNrqlJephu5mA18DjQL2lb3apmRytBcAVY2fN+RRlrkqQH6EJg1PZY7XoqGwRelXSJbspws6QDdUuqahKYtP3PXeIhumBo0Rbgou3fbN8CxoANlWuaFa0EwRngGUmrJS2hW/A5UrmmKiSJbv73Z9sf166nNtu7ba+wvYru9+Kk7QV51TcTtn8FfpG0pgwNAecrllTTZeBlSX3lvBligS6cN/GoStt/SXoH+Ipu5f8z2z9VLquWQWAYOCdpooy9Z/toxZpiftkJjJaLpgvA25XrqcL2N5IOAWfp/tvuexZoq4m0mIiIaFwrU0MREfEfEgQREY1LEERENC5BEBHRuARBRETjEgQRc0jSpnQ4jfkmQRAR0bgEQcRdSNom6bSkCUn7y/MKbkj6pPSnPyHp0bLtOklfS/pR0uHSowZJT0s6LukHSWclPVV2P9DT73+0fGs1opoEQcQ0kp4F3gAGba8DbgNvAf3At7afA8aBD8qPfA68a3stcK5nfBT41PbzdD1qrpbx9cAuumdjPEn3be+IappoMRHxPw0BLwJnysX6g8AUXZvqg2WbA8BY6d+/zPZ4GR8BvpD0ELDc9mEA2zcByv5O254s7yeAVcCp2T+siLtLEETcScCI7d3/GpTen7bdvfZn+bPn9W1yHkZlmRqKuNMJYKukxwAkPSLpCbrzZWvZ5k3glO3rwDVJG8v4MDBenv42Kem1so+lkvrm9CgiZihXIhHT2D4vaQ9wTNIi4Bawg+4hLS+Vz6bo1hEAtgP7yh/63m6dw8B+SR+Wfbw+h4cRMWPpPhoxQ5Ju2B6oXUfE/ZapoYiIxuWOICKicbkjiIhoXIIgIqJxCYKIiMYlCCIiGpcgiIho3N9CKG8NFfSEVwAAAABJRU5ErkJggg==\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "HOExaunw-Fh3",
        "outputId": "55762ae1-249b-4db3-856c-7ca462f57f1a"
      },
      "source": [
        "train_predictions = resnet50.predict(X_train)\n",
        "test_predictions = resnet50.predict(X_test)\n",
        "\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "print(\"Train Data: \"+str(accuracy_score(y_train.argmax(axis=-1), train_predictions.argmax(axis=-1))))\n",
        "print(\"Test Data: \"+str(accuracy_score(y_test.argmax(axis=-1), test_predictions.argmax(axis=-1))))"
      ],
      "execution_count": 53,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train Data: 0.5225\n",
            "Test Data: 0.21\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "OLU-A4y_4nm6"
      },
      "source": [
        "## ResNet50V2"
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Tszm9TaT5E-Q",
        "outputId": "c5ed9a69-ec0e-4d59-d753-6f7344e075a7"
      },
      "source": [
        "# ResNet50V2\n",
        "resnet50v2 = applications.ResNet50V2(weights='imagenet', include_top=False)\n",
        "\n",
        "# Add a Global Spatial Average Pooling Layer\n",
        "x = resnet50v2.output\n",
        "x = GlobalAveragePooling2D()(x)\n",
        "# Add a Fully Connected Layer\n",
        "x = Dense(512, activation='relu')(x)\n",
        "# And a Fully Connected Output/Classification Layer\n",
        "predictions = Dense(5, activation='softmax')(x)\n",
        "# Create the Full Network so we can train on it\n",
        "resnet50v2 = Model(inputs=resnet50v2.input, outputs=predictions)"
      ],
      "execution_count": 54,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
            "94674944/94668760 [==============================] - 1s 0us/step\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "JRkE4t3v5Hfs"
      },
      "source": [
        "# Compile\n",
        "resnet50v2.compile(loss='categorical_crossentropy',\n",
        "              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),\n",
        "              metrics=['accuracy'])"
      ],
      "execution_count": 55,
      "outputs": []
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Blt2niAb5Jfg",
        "outputId": "aaf16748-1733-4dba-e445-80b87c3cf47b"
      },
      "source": [
        "history_resnet50v2 = resnet50v2.fit(X_train, y_train, epochs=5, shuffle = True, verbose = 1, validation_split=0.2)"
      ],
      "execution_count": 56,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Epoch 1/5\n",
            "10/10 [==============================] - 76s 7s/step - loss: 1.7948 - accuracy: 0.2182 - val_loss: 3.4000 - val_accuracy: 0.1500\n",
            "Epoch 2/5\n",
            "10/10 [==============================] - 67s 7s/step - loss: 1.6282 - accuracy: 0.2678 - val_loss: 2.5206 - val_accuracy: 0.1750\n",
            "Epoch 3/5\n",
            "10/10 [==============================] - 66s 7s/step - loss: 1.4363 - accuracy: 0.3948 - val_loss: 2.4694 - val_accuracy: 0.0750\n",
            "Epoch 4/5\n",
            "10/10 [==============================] - 67s 7s/step - loss: 1.2267 - accuracy: 0.5436 - val_loss: 2.3561 - val_accuracy: 0.0875\n",
            "Epoch 5/5\n",
            "10/10 [==============================] - 67s 7s/step - loss: 1.0764 - accuracy: 0.6641 - val_loss: 2.2688 - val_accuracy: 0.1000\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 573
        },
        "id": "ox0-oNcQ-LGP",
        "outputId": "8b5f2810-ae05-474d-d0f5-288d854a2b69"
      },
      "source": [
        "import matplotlib.pyplot as plt\n",
        "# summarize history for accuracy\n",
        "plt.plot(history_resnet50v2.history['accuracy'], label='Training Accuracy')\n",
        "plt.plot(history_resnet50v2.history['val_accuracy'], label='Validation Accuracy')\n",
        "plt.title('Accuracy')\n",
        "plt.ylabel('Accuracy(Training and Validation)')\n",
        "plt.xlabel('epoch')\n",
        "plt.legend(loc='lower right')\n",
        "#plt.legend(['Training'],['Validation'], loc='upper left')\n",
        "plt.title('Training and Validation Accuracy')\n",
        "\n",
        "#plt.legend(['Pretrained'], loc='upper left')\n",
        "plt.show()\n",
        "# summarize history for loss\n",
        "plt.plot(history_resnet50v2.history['loss'], label='Training Loss')\n",
        "plt.plot(history_resnet50v2.history['val_loss'], label='Validation Loss')\n",
        "plt.title('Loss')\n",
        "plt.ylabel('Loss(Training and Validation)')\n",
        "plt.xlabel('epoch')\n",
        "#plt.legend(loc='upper right')\n",
        "plt.title('Training and Validation Loss')\n",
        "#plt.legend(loc='upper right')\n",
        "plt.legend(loc='upper right')\n",
        "plt.show()"
      ],
      "execution_count": 57,
      "outputs": [
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        },
        {
          "output_type": "display_data",
          "data": {
            "image/png": "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\n",
            "text/plain": [
              "<Figure size 432x288 with 1 Axes>"
            ]
          },
          "metadata": {
            "tags": [],
            "needs_background": "light"
          }
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GT8MXtEX-NA2",
        "outputId": "52c313dc-3f24-4ce9-9253-b02e31531af5"
      },
      "source": [
        "train_predictions = resnet50v2.predict(X_train)\n",
        "test_predictions = resnet50v2.predict(X_test)\n",
        "\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "\n",
        "print(\"Train Data: \"+str(accuracy_score(y_train.argmax(axis=-1), train_predictions.argmax(axis=-1))))\n",
        "print(\"Test Data: \"+str(accuracy_score(y_test.argmax(axis=-1), test_predictions.argmax(axis=-1))))"
      ],
      "execution_count": 58,
      "outputs": [
        {
          "output_type": "stream",
          "text": [
            "Train Data: 0.1625\n",
            "Test Data: 0.12\n"
          ],
          "name": "stdout"
        }
      ]
    },
    {
      "cell_type": "code",
      "metadata": {
        "id": "sBjwfLZn98p8"
      },
      "source": [
        "# Save the Model\n",
        "resnet50.save(\"/content/drive/MyDrive/resnet50.h5\")"
      ],
      "execution_count": null,
      "outputs": []
    }
  ]
}