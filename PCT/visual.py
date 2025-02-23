import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import seaborn as sns

def visualize(items,bin_size: list) -> None:
    """
    Visualize the generated items in a 3D plot.
    """
    def plot_box(ax, x0: int, y0: int, z0: int, dx: int, dy: int, dz: int, color) -> None:
        vertices = [
            [x0, y0, z0], [x0 + dx, y0, z0], [x0 + dx, y0 + dy, z0], [x0, y0 + dy, z0],
            [x0, y0, z0 + dz], [x0 + dx, y0, z0 + dz], [x0 + dx, y0 + dy, z0 + dz], [x0, y0 + dy, z0 + dz]
        ]
            
        faces = [
            [vertices[j] for j in [0, 1, 5, 4]],
            [vertices[j] for j in [7, 6, 2, 3]],
            [vertices[j] for j in [0, 3, 7, 4]],
            [vertices[j] for j in [1, 2, 6, 5]],
            [vertices[j] for j in [0, 1, 2, 3]],
            [vertices[j] for j in [4, 5, 6, 7]]
        ]
            
        ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=.3, edgecolors='k', alpha=.5, zsort='min'))

    if not items:
        raise ValueError('Items have not been generated yet')

    fig = plt.figure(figsize=(9, 5))
    ax = fig.add_subplot(111, projection='3d')

    # Create a color palette for items
    colors = sns.color_palette("pastel", len(items))

    for i, (origin, item) in enumerate(items):
        x0, y0, z0 = origin
        dx, dy, dz = item
        color = colors[i % len(colors)]
        plot_box(ax, x0, y0, z0, dx, dy, dz, color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set limits for the axes
    ax.set_xlim([0, bin_size[0]])
    ax.set_ylim([0, bin_size[1]])
    ax.set_zlim([0, bin_size[2]])

    ax.title.set_text(f'3D Bin Packing Visualization')
    ax.set_box_aspect([bin_size[0], bin_size[1], bin_size[2]])
        
    # Add a legend with information
    total_volume = sum([item[0]*item[1]*item[2] for _, (origin, item) in enumerate(items)])
    volume_per_item = total_volume/(bin_size[0]*bin_size[1]*bin_size[2])
    info_text = (
        f'Bin size: {bin_size}\n'
        f'Number of bins: {len(items)}\n'
        f'Number of items per bin: {volume_per_item}\n'
        f'Total volume of items: {total_volume}'
    )
    plt.figtext(.77, .5, info_text, fontsize=8, ha='left', va='center', bbox=dict(facecolor='white', edgecolor='black'))

    plt.show()

if __name__ == '__main__':
    visualize(items= [([1,2,3],[14,15,26]),([3,4,5],[12,12,14]),([3,4,5],[13,13,14])], bin_size= [100,100,100])