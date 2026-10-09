import numpy as np
import matplotlib.pyplot as plt
import os
import time
from matplotlib.ticker import FormatStrFormatter

def show_2Dimages(images: list, names: list = None, title: str = None, shape: tuple = None, axis = 'off', save_path: str = None, save_name: str = "file_name", show = True):
    plt.close()    
    images = np.array(images)
    cnt = len(images)
    
    if cnt == 1:
        if len(images[0].shape) == 2 or images[0].shape[0] == 1:
            plt.imshow(images[0], cmap = "gray")
        else:
            plt.imshow(images[0].transpose(1, 2, 0))
        if names != None: plt.title(names[0])
        plt.axis(axis)  
         
    else:
        row = 1
        col = cnt
        weight = 4 * cnt
        height = 5
        
        if shape != None:
            row = shape[0]
            col = shape[1]
            weight = 3 * col 
            height = 4 * row 
        
        fig, axes = plt.subplots(row, col, figsize = (weight, height))
        
        if row == 1 or col == 1:
            for i, image in enumerate(images):
                if len(image.shape) == 2 or images[0].shape[0] == 1:
                    axes[i].imshow(image, cmap = "gray")
                else:
                    axes[i].imshow(image.transpose(1, 2, 0))
                
                if names != None: axes[i].set_title(names[i])
                axes[i].axis(axis)
        else:
            i = 0
            for r in range(row):
                for c in range(col):
                    if (r * col + c + 1) <= cnt: 
                        if len(images[i].shape) == 2 or images[0].shape[0] == 1:
                            axes[r][c].imshow(images[i], cmap = "gray")
                        else:
                            axes[r][c].imshow(images.transpose(1, 2, 0))
                        
                        if names != None: axes[r][c].set_title(names[i])
                    axes[r][c].axis(axis)
                    i += 1
        
    if title != None: fig.suptitle(title)
    if save_path != None: 
        current_time = f"{time.time_ns()}"
        if not os.path.exists(save_path): os.makedirs(save_path)
        if cnt == 1: plt.savefig("{}_{}.png".format(os.path.join(save_path, save_name), current_time), dpi = 300)
        else: fig.savefig("{}_{}.png".format(os.path.join(save_path, save_name), current_time), dpi = 300)
    
    if show: plt.show()
    
    
def boxplot_Compare_Two_Pic(arr_cal, arr_real, names = None, title = None, xlabel = None, ylabel = None, show = True, save_path = "./", save_name: str = "file_name", theoretical_value = None, idx = None):
    plt.close()
    if names == None:
        names = ['data1', 'data2']
    if title == None:
        title = 'boxplot'
    if xlabel == None:
        xlabel = "images"
    if ylabel == None:
        ylabel = "value"
    
    mid = []
        

    arr_cal = arr_cal.flatten()
    arr_real = arr_real.flatten()


    fig, ax = plt.subplots()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    boxplot = ax.boxplot([arr_cal, arr_real], labels=names)

    small_font = {'size': 8}


    for i, box in enumerate(boxplot['boxes']):

        x1, y1 = box.get_xdata()[0], box.get_ydata()[0]
        x2, y2 = box.get_xdata()[2], box.get_ydata()[2]
        q1 = y1
        q3 = y2
        median = boxplot['medians'][i].get_ydata()[0]
        whisker_min = boxplot['whiskers'][2 * i].get_ydata()[1]
        whisker_max = boxplot['whiskers'][2 * i + 1].get_ydata()[1]
        mid.append(median)

        ax.text(0.9, q1, f'{q1:.6f} :Q1', verticalalignment='top', horizontalalignment = 'right', color='g')
        ax.text(0.9, q3, f'{q3:.6f} :Q3', verticalalignment='bottom', horizontalalignment = 'right', color='g')
        ax.text(i + 1.1, median, f'Median: {median:.6f}', fontdict=small_font)


    title = f"{title} | Median difference: {abs(mid[0]-mid[1]):.6f}"
    if theoretical_value != None: title = f"{title}--The theoretical_value is {theoretical_value[idx]:.6f}"
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    if show: plt.show()
    if save_path != None:
        current_time = f"{time.time_ns()}"
        if not os.path.exists(save_path): os.makedirs(save_path)
        fig.savefig("{}_{}.png".format(os.path.join(save_path, save_name), current_time), dpi = 300)
    
    
   
def boxplot_Compare_Stdval(arrs, std, shape = None, names = None, ylabels = None, title = None, save_path: str = "./", save_name = "file_name", show = True):

    plt.close()
    cnt = len(arrs)
    
    if title == None: title = "Boxplot"
    if names == None: names = [f"Data {x}" for x in range(1, cnt + 1)]
    if ylabels == None: ylabels = [f"Value" for x in range(1, cnt + 1)]
    
    if cnt == 1:
        data = arrs[0].flatten()
        plt.boxplot(data, vert=True)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        
        median = np.median(data)
        q1 = np.percentile(data, 25)
        q3 = np.percentile(data, 75)


        plt.text(1.1, median, f'Median: {median:.6f}', verticalalignment='center', color='r')
        plt.text(0.9, q1, f'{q1:.6f} :Q1', verticalalignment='top', horizontalalignment = 'right', color='g')
        plt.text(0.9, q3, f'{q3:.6f} :Q3', verticalalignment='bottom', horizontalalignment = 'right', color='g')

        plt.xticks([])
        plt.xlabel(f'{names[0]} \n Median:{median: .6f} | distance:{abs(std - median): .6f}')

    else:
        
        row = 1
        col = cnt
        weight = 6 * cnt
        height = 6
        
        if shape != None:
            row = shape[0]
            col = shape[1]
            weight = 6 * col 
            height = 8 * row 
            
        fig, ax = plt.subplots(row, col, figsize = (weight, height))
        
        if row == 1:
            for i in range(0, cnt):
                data = arrs[i].flatten()

                ax[i].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                ax[i].boxplot(data, vert=True)

                median = np.median(data)
                q1 = np.percentile(data, 25)
                q3 = np.percentile(data, 75)

                ax[i].text(1.1, median, f'Median: {median:.6f}', verticalalignment='center', color='r')
                ax[i].text(0.9, q1, f'{q1:.6f} :Q1', verticalalignment='top', horizontalalignment = 'right', color='g')
                ax[i].text(0.9, q3, f'{q3:.6f} :Q3', verticalalignment='bottom', horizontalalignment = 'right', color='g')

                ax[i].set_title(f'{names[i]} \n Median:{median: .6f} | distance:{abs(std - median): .6f}')

                ax[i].set_xticks([])
        else:
            i = 0
            for r in range(row):
                for c in range(col):
                    if (r * col + c + 1) > cnt: 
                        ax[r][c].axis("off")
                        break
                    data = arrs[i].flatten()

                    ax[r][c].yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
                    ax[r][c].boxplot(data, vert=True)

                    median = np.median(data)
                    q1 = np.percentile(data, 25)
                    q3 = np.percentile(data, 75)

                    ax[r][c].text(1.1, median, f'Median: {median:.6f}', verticalalignment='center', color='r')
                    ax[r][c].text(0.9, q1, f'{q1:.6f} :Q1', verticalalignment='top', horizontalalignment = 'right', color='g')
                    ax[r][c].text(0.9, q3, f'{q3:.6f} :Q3', verticalalignment='bottom', horizontalalignment = 'right', color='g')

                    ax[r][c].set_title(f'{names[i]} \n Median:{median: .6f} | distance:{abs(std - median): .6f}')

                    ax[r][c].set_xticks([])
                    i += 1

    proTitle = f"{title} | std: {std}"
    if cnt == 1: plt.title(proTitle)
    else: fig.suptitle(proTitle)
    
    if save_path != None: 
        current_time = f"{time.time_ns()}"
        if not os.path.exists(save_path): os.makedirs(save_path)
        if cnt == 1: plt.savefig("{}_{}.png".format(os.path.join(save_path, save_name), current_time), dpi = 300)
        else: fig.savefig("{}_{}.png".format(os.path.join(save_path, save_name), current_time), dpi = 300)
    
    if show == True: plt.show()
    
    
    
def show_lines(datas: list[1], x_labels: list = [], y_labels: list = [], subtitle: list = [], title: str = None, shape: tuple = None,  save_path: str = None, save_name: str = "file_name", show = True):

    plt.close()
    datas = np.array(datas)
    cnt = len(datas)
    
    while len(x_labels) < cnt:
        x_labels.append("X value")
        
    while len(y_labels) < cnt:
        y_labels.append("Y value")
        
    idx = 1
    while len(subtitle) < cnt:
        subtitle.append(f"line {idx}")
        idx += 1
    
    if cnt == 1:
        plt.plot(list(range(len(datas[0]))), datas[0], 'b-', linewidth = 1)
        if title != None: plt.title(title)
        plt.xlabel(x_labels[0])
        plt.ylabel(y_labels[0])
        plt.grid(True)
         
    else:
        row = 1
        col = cnt
        weight = 5 * cnt
        height = 4
        
        if shape != None:
            row = shape[0]
            col = shape[1]
            weight = 5 * col 
            height = 4 * row 
        
        fig, axes = plt.subplots(row, col, figsize = (weight, height))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        if row == 1 or col == 1:
            for i, data in enumerate(datas):
                axes[i].plot(list(range(len(data))), data, 'b-', linewidth = 1)
                axes[i].set_title(subtitle[i])
                axes[i].set_xlabel(x_labels[i])
                axes[i].set_ylabel(y_labels[i])
                axes[i].grid(True)

        else:
            i = 0
            for r in range(row):
                for c in range(col):
                    if (r * col + c + 1) <= cnt: 
                        axes[r][c].plot(list(range(len(datas[i]))), datas[i], 'b-', linewidth = 1)
                        axes[r][c].set_title(subtitle[i])
                        axes[r][c].set_xlabel(x_labels[i])
                        axes[r][c].set_ylabel(y_labels[i])
                        axes[r][c].grid(True)
                    else: axes[r][c].axis('off')
                    i += 1
        
        if title != None: fig.suptitle(title)
    if save_path != None: 
        current_time = f"{time.time_ns()}"
        if not os.path.exists(save_path): os.makedirs(save_path)
        if cnt == 1: plt.savefig("{}_{}.png".format(os.path.join(save_path, save_name), current_time), dpi = 300)
        else: fig.savefig("{}_{}.png".format(os.path.join(save_path, save_name), current_time), dpi = 300)
    
    if show: plt.show()
    
    
    
def drawline(data:list, save_path, save_name, text, loss_name = None):
    colors = ['blue', 'red', 'green', 'black', "yellow"]
    linestyles = ['-', '-', '-', '-', '-']
    labels = ['S', 'S ema', 'fusion', 'T', "loss"]
    # labels = ['S','T']
    # linestyles = ['-', '-']
    # colors = ['blue', 'red']
    
    marker = 'o'
    markersize = '3'
    x = np.arange(2000, len(data[0][0]) * 2000 + 1, 2000)
    
    if loss_name == None: fig, axes = plt.subplots(1, 3, figsize=(24, 6))
    else: fig, axes = plt.subplots(1, 4, figsize=(30, 6))
        

    for i, y in enumerate(data[0]):
        axes[0].plot(x, y, label=labels[i], color=colors[i], 
                    linestyle=linestyles[i], linewidth=1, marker=marker, markersize=markersize)
    min_idx = np.argmin(data[0][0])
    min_x = x[min_idx]
    min_y = data[0][0][min_idx]
    axes[0].scatter(
        min_x, min_y,
        marker='*',  
        color='black',  
        s=220,  
        edgecolor='white',  
        linewidth=1.5,
        zorder=10  
    )
    axes[0].set_xlim(0, max(x)*1.05)
    axes[0].set_title(f'RMSE - highest point: ({min_y:.4f}, {min_x:.0f})', fontsize=16)
    axes[0].set_ylabel('value', fontsize=14)
    axes[0].grid(True, linestyle='--', alpha=0.7)
    axes[0].legend(fontsize=12)

    for i, y in enumerate(data[1]):
        axes[1].plot(x, y, label=labels[i], color=colors[i], 
                    linestyle=linestyles[i], linewidth=1, marker=marker, markersize=markersize)
    max_idx = np.argmax(data[1][0])
    max_x = x[max_idx]
    max_y = data[1][0][max_idx]
    axes[1].scatter(
        max_x, max_y,
        marker='*',  
        color='black', 
        s=220,  
        edgecolor='white',  
        linewidth=1.5,
        zorder=10  
    )
    axes[1].set_xlim(0, max(x)*1.05)
    axes[1].set_title(f'PSNR - highest point: ({max_y:.4f}, {max_x:.0f})', fontsize=16)
    axes[1].set_ylabel('value', fontsize=14)
    axes[1].grid(True, linestyle='--', alpha=0.7)
    axes[1].legend(fontsize=12)
    # axes[1].set_xlim(0, 10)

    for i, y in enumerate(data[2]):
        axes[2].plot(x, y, label=labels[i], color=colors[i], 
                    linestyle=linestyles[i], linewidth=1, marker=marker, markersize=markersize)
    max_idx = np.argmax(data[2][0])
    max_x = x[max_idx]
    max_y = data[2][0][max_idx]
    axes[2].scatter(
        max_x, max_y,
        marker='*', 
        color='black',  
        s=220,  
        edgecolor='white',  
        linewidth=1.5,
        zorder=10  
    )
    axes[2].set_xlim(0, max(x)*1.05)
    axes[2].set_title(f'SSIM - highest point: ({max_y:.4f}, {max_x:.0f})', fontsize=16)
    axes[2].set_ylabel('value', fontsize=14)
    
    axes[2].grid(True, linestyle='--', alpha=0.7)
    axes[2].legend(fontsize=12)

    if loss_name is not None:
        
        for i, y in enumerate(data[3]):
            axes[3].plot(x, y, label=labels[i], color=colors[i], 
                        linestyle=linestyles[i], linewidth=1, marker=marker, markersize=markersize)

        axes[3].set_xlim(0, max(x)*1.05)
        axes[3].set_title(loss_name, fontsize=16)
        axes[3].set_ylabel('value', fontsize=14)
        
        axes[3].grid(True, linestyle='--', alpha=0.7)
        axes[3].legend(fontsize=12)

    fig.suptitle(text)

    plt.tight_layout()
    fig.savefig(os.path.join(save_path, f"{save_name}.png"), dpi = 300)    
    
    

if __name__ == "__main__":
    a = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 2, 1, 3, 4, 5]
    b = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 2, 1, 3, 4, 5]
    c = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 2, 1, 3, 4, 5]
    d = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 2, 1, 3, 4, 5]
    e = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 2, 1, 3, 4, 5]
    f = [1, 2, 3, 4, 5, 6, 7, 6, 5, 4, 2, 1, 3, 4, 5]
    show_lines([a, b, c, d, e], x_labels=["epoch"], y_labels=["mse"], title = "MSE", save_path="./")