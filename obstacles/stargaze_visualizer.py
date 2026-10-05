import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
import numpy as np
import random

class ConstellationVisualizer:
    def __init__(self):
        # Layer configurations: Distance between each nested cube (15, 30, 45)
        # L represents the half-length of the cube edges
        self.L_values = [15, 30, 45] 
        # Colors for innermost (Layer 1), middle (Layer 2), and outermost (Layer 3)
        self.layer_colors = ['#00FF00', '#FF9900', '#0099FF'] 
        self.obstacle_points = [] # Stores coordinates of all random points
        self.path_artists = [] # Stores matplotlib objects (lines, texts, scatter) to easily remove them later
        
        # Setup Figure and 3D Axis
        self.fig = plt.figure(figsize=(14, 10))
        self.fig.canvas.manager.set_window_title('3D Dynamic Maximum Clearance Path Visualization')
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Adjust layout spacing: 3D plot takes the left side (up to 60%), controls take the right side
        self.fig.subplots_adjust(bottom=0.05, left=0.05, right=0.60, top=0.95) 
        
        # Define face options for the UI
        self.faces_options = ['Top (Z+)', 'Bottom (Z-)', 'Right (X+)', 'Left (X-)', 'Front (Y+)', 'Back (Y-)']
        
        # --- UI SETUP (ALL ELEMENTS ON THE RIGHT PANEL) ---
        
        # 1. UI: Info Panel (Top-Right) for Top 5 Closest Points
        self.ax_info = plt.axes([0.65, 0.60, 0.30, 0.35])
        self.ax_info.axis('off') # Hide border and ticks
        self.info_text = self.ax_info.text(0, 1, "Top 5 Closest Points:\n(Calculate path first)", 
                                           va='top', fontsize=10, family='monospace')
        
        # 2. UI: Titles for Radio Buttons (Middle-Right)
        self.fig.text(0.65, 0.56, "1. Initial Search Face", weight='bold', fontsize=10)
        self.fig.text(0.81, 0.56, "2. Forbidden Face", weight='bold', fontsize=10, color='red')
        
        # 3. UI: Radio Buttons (Select Initial Search Face)
        self.ax_radio_start = plt.axes([0.65, 0.36, 0.14, 0.18])
        self.radio_start = RadioButtons(self.ax_radio_start, self.faces_options, activecolor='green')
        self.current_start = self.faces_options[0]
        self.radio_start.on_clicked(self.set_start)
        
        # 4. UI: Radio Buttons (Select Forbidden Face)
        self.forbid_options = ['None'] + self.faces_options
        self.ax_radio_forbid = plt.axes([0.81, 0.33, 0.14, 0.21]) # Slightly taller for 7 options
        self.radio_forbid = RadioButtons(self.ax_radio_forbid, self.forbid_options, activecolor='red')
        self.current_forbid = self.forbid_options[0]
        self.radio_forbid.on_clicked(self.set_forbid)
        
        # 5. UI: Action Buttons (Bottom-Right)
        self.ax_btn_random = plt.axes([0.65, 0.20, 0.30, 0.08])
        self.btn_random = Button(self.ax_btn_random, '1. Randomize Constellation')
        self.btn_random.on_clicked(self.randomize_points)
        
        self.ax_btn_path = plt.axes([0.65, 0.10, 0.30, 0.08])
        self.btn_path = Button(self.ax_btn_path, '2. Find Max Clearance Path', color='cyan')
        self.btn_path.on_clicked(self.find_path)
        
        # Initial render on startup
        self.randomize_points(None)
        plt.show()

    # --- CALLBACK FUNCTIONS ---
    def set_start(self, label):
        """Updates the selected initial search face."""
        self.current_start = label

    def set_forbid(self, label):
        """Updates the selected forbidden face."""
        self.current_forbid = label

    # --- CORE LOGIC FUNCTIONS ---
    def generate_points(self):
        """
        Generates the 3D point cloud based on the specific rules:
        - 1 Core point at (0,0,0)
        - 3 to 5 points distributed across 3 layers per direction.
        - No points occupy the same segment.
        - Points are placed randomly within their segments (not centroid).
        """
        xs, ys, zs, colors, sizes = [], [], [], [], []
        self.obstacle_points = []
        
        # Nucleus / Anchor Point at the center
        xs.append(0); ys.append(0); zs.append(0)
        colors.append('red'); sizes.append(150)
        
        # 6 Directions (Faces of the cube)
        faces = [('z', 1), ('z', -1), ('x', 1), ('x', -1), ('y', 1), ('y', -1)]
        
        for axis, sign in faces:
            # Rule: 3 to 5 points randomly scattered across the 3 layers for this face direction
            total_points = random.randint(3, 5)
            
            # Distribute points to layers without leaving any layer empty
            if total_points == 3: p = [1, 1, 1]
            elif total_points == 4: p = random.choice([[2,1,1], [1,2,1], [1,1,2]])
            else: p = random.choice([[3,1,1], [1,3,1], [1,1,3], [2,2,1], [2,1,2], [1,2,2]])
                
            for layer_idx in range(3):
                L = self.L_values[layer_idx]
                num_points = p[layer_idx]
                
                # 3x3 grid per face = 9 segments. Pick unique segments to avoid overlapping.
                segments = [(i, j) for i in range(3) for j in range(3)]
                chosen_segments = random.sample(segments, num_points)
                segment_length = (2 * L) / 3
                
                for i, j in chosen_segments:
                    # Calculate min/max boundaries of the chosen segment
                    u_min = -L + (i * segment_length)
                    u_max = -L + ((i + 1) * segment_length)
                    v_min = -L + (j * segment_length)
                    v_max = -L + ((j + 1) * segment_length)
                    
                    # Add 10% margin so points don't stick exactly to the segment grid lines
                    margin = segment_length * 0.1
                    u = random.uniform(u_min + margin, u_max - margin)
                    v = random.uniform(v_min + margin, v_max - margin)
                    w = sign * L
                    
                    # Assign the calculated local variables back to global 3D coordinates based on axis
                    if axis == 'z': x, y, z = u, v, w
                    elif axis == 'y': x, y, z = u, w, v
                    elif axis == 'x': x, y, z = w, u, v
                        
                    xs.append(x); ys.append(y); zs.append(z)
                    colors.append(self.layer_colors[layer_idx])
                    sizes.append(40)
                    self.obstacle_points.append([x, y, z]) # Store for path calculation
                    
        return xs, ys, zs, colors, sizes

    def draw_transparent_cubes(self):
        """Draws the 3 nested 3x3 grid wireframe cubes to serve as spatial guides."""
        for L in self.L_values:
            steps = np.linspace(-L, L, 4)
            for y in steps:
                for z in [-L, L]: self.ax.plot([-L, L], [y, y], [z, z], color='gray', alpha=0.15, linewidth=1)
            for z in steps:
                for y in [-L, L]: self.ax.plot([-L, L], [y, y], [z, z], color='gray', alpha=0.15, linewidth=1)
            for x in steps:
                for z in [-L, L]: self.ax.plot([x, x], [-L, L], [z, z], color='gray', alpha=0.15, linewidth=1)
            for z in steps:
                for x in [-L, L]: self.ax.plot([x, x], [-L, L], [z, z], color='gray', alpha=0.15, linewidth=1)
            for x in steps:
                for y in [-L, L]: self.ax.plot([x, x], [y, y], [-L, L], color='gray', alpha=0.15, linewidth=1)
            for y in steps:
                for x in [-L, L]: self.ax.plot([x, x], [y, y], [-L, L], color='gray', alpha=0.15, linewidth=1)

        # Draw Axis Labels at the corners (edges) of the outermost cube
        L_max = self.L_values[-1]
        
        # Position texts at the corner (-L, -L, -L) extending outwards along the edges
        self.ax.text(L_max + 5, -L_max, -L_max, 'X-Axis', color='black', weight='bold')
        self.ax.text(-L_max, L_max + 5, -L_max, 'Y-Axis', color='black', weight='bold')
        self.ax.text(-L_max, -L_max, L_max + 5, 'Z-Axis', color='black', weight='bold')

    def draw_face_surface(self, face_name, color, alpha):
        """Draws a semi-transparent colored square on a specified face (used to show forbidden/search areas)."""
        L = self.L_values[-1]
        F_grid = np.array([-L, L])
        F1, F2 = np.meshgrid(F_grid, F_grid)
        
        if face_name == 'Top (Z+)': F3 = np.full_like(F1, L); surf = self.ax.plot_surface(F1, F2, F3, color=color, alpha=alpha)
        elif face_name == 'Bottom (Z-)': F3 = np.full_like(F1, -L); surf = self.ax.plot_surface(F1, F2, F3, color=color, alpha=alpha)
        elif face_name == 'Right (X+)': F3 = np.full_like(F1, L); surf = self.ax.plot_surface(F3, F1, F2, color=color, alpha=alpha)
        elif face_name == 'Left (X-)': F3 = np.full_like(F1, -L); surf = self.ax.plot_surface(F3, F1, F2, color=color, alpha=alpha)
        elif face_name == 'Front (Y+)': F3 = np.full_like(F1, L); surf = self.ax.plot_surface(F1, F3, F2, color=color, alpha=alpha)
        elif face_name == 'Back (Y-)': F3 = np.full_like(F1, -L); surf = self.ax.plot_surface(F1, F3, F2, color=color, alpha=alpha)
        return surf

    def get_face_points(self, face_name, L, resolution):
        """Generates a grid of candidate points (thousands of them) on a specific cube face."""
        grid = np.linspace(-L, L, resolution)
        G1, G2 = np.meshgrid(grid, grid)
        G1 = G1.flatten(); G2 = G2.flatten()
        M = len(G1)
        S = np.zeros((M, 3))
        
        if face_name == 'Top (Z+)': S[:,0]=G1; S[:,1]=G2; S[:,2]=L
        elif face_name == 'Bottom (Z-)': S[:,0]=G1; S[:,1]=G2; S[:,2]=-L
        elif face_name == 'Right (X+)': S[:,0]=L; S[:,1]=G1; S[:,2]=G2
        elif face_name == 'Left (X-)': S[:,0]=-L; S[:,1]=G1; S[:,2]=G2
        elif face_name == 'Front (Y+)': S[:,0]=G1; S[:,1]=L; S[:,2]=G2
        elif face_name == 'Back (Y-)': S[:,0]=G1; S[:,1]=-L; S[:,2]=G2
        return S

    def identify_face(self, S_point, L):
        """Helper to determine which face a final coordinate landed on."""
        if np.isclose(S_point[2], L): return 'Top (Z+)'
        if np.isclose(S_point[2], -L): return 'Bottom (Z-)'
        if np.isclose(S_point[0], L): return 'Right (X+)'
        if np.isclose(S_point[0], -L): return 'Left (X-)'
        if np.isclose(S_point[1], L): return 'Front (Y+)'
        if np.isclose(S_point[1], -L): return 'Back (Y-)'
        return 'Unknown'

    def clear_old_paths(self):
        """Removes the old calculated line, start point, texts, and highlights before drawing a new one."""
        for artist in self.path_artists:
            try: artist.remove()
            except: pass
        self.path_artists.clear()

    # --- MAIN ACTIONS ---
    def randomize_points(self, event):
        """Triggered by 'Randomize' button. Resets the 3D space with a new constellation."""
        self.ax.cla()
        self.ax.axis('off') 
        self.ax.set_box_aspect([1, 1, 1])
        
        # Set bounds slightly larger than the outer cube
        limit = self.L_values[-1] + 15
        self.ax.set_xlim([-limit, limit]); self.ax.set_ylim([-limit, limit]); self.ax.set_zlim([-limit, limit])
        
        self.draw_transparent_cubes()
        xs, ys, zs, colors, sizes = self.generate_points()
        
        # Draw the points
        self.ax.scatter(xs[0], ys[0], zs[0], c=colors[0], s=sizes[0], marker='*', edgecolors='black') # Nucleus
        self.ax.scatter(xs[1:], ys[1:], zs[1:], c=colors[1:], s=sizes[1:], alpha=1.0, edgecolors='black') # Layer points
        
        self.clear_old_paths()
        self.info_text.set_text("Top 5 Closest Points:\n(Calculate path first)") # Reset text
        self.ax.set_title("Waiting for Calculation...", fontsize=12)
        self.fig.canvas.draw_idle()

    def find_path(self, event):
        """
        The core algorithm: 
        1. Selects valid faces to search.
        2. Generates thousands of candidates.
        3. Uses Vectorized Cross-Product to find distance from all candidate lines to all obstacles.
        4. Selects the line with the maximum minimum-distance (Maximum Clearance).
        5. Identifies and lists the top 5 closest points to this winning path.
        """
        self.clear_old_paths()
        
        L = self.L_values[-1]
        resolution = 80 # 80x80 grid per face = 6,400 points per face tested simultaneously
        
        opposites = {
            'Top (Z+)': 'Bottom (Z-)', 'Bottom (Z-)': 'Top (Z+)',
            'Right (X+)': 'Left (X-)', 'Left (X-)': 'Right (X+)',
            'Front (Y+)': 'Back (Y-)', 'Back (Y-)': 'Front (Y+)'
        }
        
        # Determine valid search faces (Initial face + 4 adjacent faces. Opposite face is ignored).
        search_faces = [f for f in self.faces_options if f != opposites[self.current_start]]
        
        # Remove the forbidden face from the search pool
        if self.current_forbid in search_faces:
            search_faces.remove(self.current_forbid)
            
        if len(search_faces) == 0:
            self.ax.set_title("⚠️ Error: All faces are blocked!")
            self.fig.canvas.draw_idle()
            return

        # Stack all candidate points into one large Array (S_valid)
        S_list = [self.get_face_points(f, L, resolution) for f in search_faces]
        S_valid = np.vstack(S_list) 
        P = np.array(self.obstacle_points)
        
        # --- VECTORIZED RAY CASTING ALGORITHM ---
        # Formula: Distance = |Point x Start| / |Start|
        # This calculates perpendicular distance from all Obstacles (P) to all Candidate Lines (S)
        cross_prod = np.cross(P[:, np.newaxis, :], S_valid[np.newaxis, :, :])
        cross_norm = np.linalg.norm(cross_prod, axis=2)
        S_norm = np.linalg.norm(S_valid, axis=1)
        distances = cross_norm / S_norm[np.newaxis, :]
        
        # Ignore obstacles that are "behind" the laser direction (Dot product < 0)
        dot_prod = np.sum(P[:, np.newaxis, :] * S_valid[np.newaxis, :, :], axis=2)
        distances[dot_prod < 0] = np.inf 
        
        # min_dists contains the bottleneck (closest point) for EACH candidate line
        min_dists = np.min(distances, axis=0) 
        # best_idx is the index of the line that has the LARGEST bottleneck (Max Clearance)
        best_idx = np.argmax(min_dists) 
        
        best_S = S_valid[best_idx]
        max_clearance = min_dists[best_idx]
        
        # Find which exact obstacle point acts as the bottleneck for this winning path
        idx_closest_obstacle = np.argmin(distances[:, best_idx])
        P_bottle = P[idx_closest_obstacle]
        
        # Project the bottleneck point onto the line to draw the dashed red indicator
        t = np.dot(P_bottle, best_S) / np.dot(best_S, best_S)
        projection_point = t * best_S
        
        # --- TOP 5 CLOSEST POINTS LOGIC ---
        # Get distances of all points exclusively to the winning path
        distances_to_best_line = distances[:, best_idx]
        # Sort indices from closest to farthest
        sorted_indices = np.argsort(distances_to_best_line)
        
        # Build the string for the UI right panel
        info_str = f"MAX CLEARANCE:\n{max_clearance:.2f} Units\n\n"
        info_str += "TOP 5 CLOSEST POINTS:\n"
        info_str += "-"*22 + "\n"
        
        for i in range(min(5, len(sorted_indices))):
            idx = sorted_indices[i]
            dist = distances_to_best_line[idx]
            pt = P[idx]
            
            # Format text for right panel
            info_str += f"{i+1}. Dist: {dist:.2f}\n   [X:{pt[0]:.0f}, Y:{pt[1]:.0f}, Z:{pt[2]:.0f}]\n\n"
            
            # Draw small number indicator in 3D space next to the point
            # Offset slightly (+2 to coordinates) so it doesn't cover the point itself
            rank_text = self.ax.text(pt[0]+2, pt[1]+2, pt[2]+2, str(i+1), 
                                     color='black', weight='bold', fontsize=9, zorder=10,
                                     bbox=dict(facecolor='yellow', alpha=0.8, edgecolor='black', boxstyle='round,pad=0.2'))
            # Add to path_artists so it gets cleared on the next calculation
            self.path_artists.append(rank_text)
            
        self.info_text.set_text(info_str)
        
        # --- DRAWING RESULTS ---
        # Highlight initial search face (Green)
        if self.current_start in search_faces:
            surf_start = self.draw_face_surface(self.current_start, color='green', alpha=0.1)
            self.path_artists.append(surf_start)
            
        # Highlight forbidden face (Red)
        if self.current_forbid != 'None':
            surf_forbid = self.draw_face_surface(self.current_forbid, color='red', alpha=0.3)
            self.path_artists.append(surf_forbid)

        # Draw the main laser path and start point
        line, = self.ax.plot([best_S[0], 0], [best_S[1], 0], [best_S[2], 0], color='cyan', linewidth=4, zorder=5)
        start_pt = self.ax.scatter(*best_S, color='cyan', s=100, marker='s', edgecolors='black', zorder=6)
        
        # Draw the dashed red line pointing to the absolute closest point (the bottleneck)
        bottle_line, = self.ax.plot([P_bottle[0], projection_point[0]], 
                                    [P_bottle[1], projection_point[1]], 
                                    [P_bottle[2], projection_point[2]], 
                                    color='red', linestyle='--', linewidth=2, zorder=4)
        
        self.path_artists.extend([line, start_pt, bottle_line])
        
        final_face = self.identify_face(best_S, L)
        self.ax.set_title(f"Target: {self.current_start}  |  Path Landed On: {final_face}", fontsize=12)
        self.fig.canvas.draw_idle()

if __name__ == '__main__':
    Visualizer = ConstellationVisualizer()
