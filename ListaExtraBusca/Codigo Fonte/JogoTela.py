import tkinter as tk
from tkinter import ttk, messagebox
import heapq
import math
import time
from collections import deque

TAM_CELULA = 25
CORES = {
    'PAREDE': '#2c3e50',      
    'LIVRE': '#ecf0f1',        
    'START': '#27ae60',        
    'END': '#c0392b',          
    'CAMINHO': '#2980b9',      
    'VISITADO': '#a9cce3',     
    'FRONTEIRA': '#f1c40f',    
    'TEXTO': '#000000'
}

MAPAS = [
    # Mapa 1
    """
#########################
#S                      #
# ####### # ########### #
# #       # #           #
# # ##### # # ######### #
# #     # # #           #
# ##### # # ########### #
#       # #             #
####### # # ########### #
#       #              E#
#########################
""",
    # Mapa 2
    """
#########################
#S#                     #
# # ################### #
# # #                 # #
# # # ############### # #
# # # #             # # #
# # # # ########### # # #
# # # #           # # # #
# # # ########### # # # #
#                   #  E#
#########################
""",
    # Mapa 3
    """
#########################
#S      #       #       #
####### # ##### # ##### #
#       #     # #     # #
# ####### ### # # ### # #
#         #   # #   #   #
####### ### ### ### ### #
#     #   #     #   #   #
# ### ### ####### ### # #
#   #                 #E#
#########################
""",
    # Mapa 4
    """
#########################
#S  #   #     #   #     #
# #   #   # #   #   # # #
#   #   #     #   #     #
# #   #   # #   #   # # #
#   #   #     #   #     #
# #   #   # #   #   # # #
#   #   #     #   #     #
# #   #   # #   #   # #E#
#########################
""",
    # Mapa 5
    """
#########################
#S                      #
# ##################### #
# #                   # #
# # ################# # #
# # #               # # #
# # # ############# # # #
# # # #           # # # #
# # # # ######### # # # #
#     #           #    E#
#########################
"""
]

CONFIGS_POSICOES = [
    [((1, 1), (9, 23)), ((1, 1), (1, 23)), ((9, 1), (1, 23)), ((5, 13), (9, 1)), ((1, 12), (9, 23))],
    [((1, 1), (9, 23)), ((1, 23), (9, 1)), ((5, 5), (5, 19)), ((2, 2), (8, 22)), ((9, 1), (1, 1))],
    [((1, 1), (8, 23)), ((1, 23), (8, 1)), ((4, 12), (1, 1)), ((8, 1), (1, 23)), ((5, 5), (5, 20))],
    [((1, 1), (7, 23)), ((1, 23), (7, 1)), ((1, 12), (7, 12)), ((4, 1), (4, 23)), ((2, 2), (6, 22))],
    [((1, 1), (8, 23)), ((8, 1), (1, 23)), ((1, 1), (5, 12)), ((8, 23), (4, 4)), ((2, 2), (7, 22))]
]

class MazeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visualizador de Algoritmos de Busca (Tkinter)")
        self.root.geometry("1200x700")

        self.mapa_idx = 0
        self.pos_idx = 0
        self.algoritmo_var = tk.StringVar(value="BFS")
        self.rects = {} 
        self.grid_chars = []
        
        # --- Layout Principal ---
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Painel de Controle
        control_panel = tk.Frame(main_frame, width=350, bg="#f0f0f0", relief=tk.RAISED, bd=2)
        control_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        control_panel.pack_propagate(False)

        # Widgets do Painel
        tk.Label(control_panel, text="CONFIGURAÇÕES", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Seleção de Mapa
        frame_mapa = tk.Frame(control_panel, bg="#f0f0f0")
        frame_mapa.pack(pady=5)
        tk.Button(frame_mapa, text="<", command=lambda: self.mudar_mapa(-1)).pack(side=tk.LEFT)
        self.lbl_mapa = tk.Label(frame_mapa, text="Mapa 1", bg="#f0f0f0", width=10)
        self.lbl_mapa.pack(side=tk.LEFT)
        tk.Button(frame_mapa, text=">", command=lambda: self.mudar_mapa(1)).pack(side=tk.LEFT)

        # Seleção de Posição
        frame_pos = tk.Frame(control_panel, bg="#f0f0f0")
        frame_pos.pack(pady=5)
        tk.Button(frame_pos, text="<", command=lambda: self.mudar_posicao(-1)).pack(side=tk.LEFT)
        self.lbl_pos = tk.Label(frame_pos, text="Posição 1", bg="#f0f0f0", width=10)
        self.lbl_pos.pack(side=tk.LEFT)
        tk.Button(frame_pos, text=">", command=lambda: self.mudar_posicao(1)).pack(side=tk.LEFT)

        tk.Label(control_panel, text="ALGORITMO", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        algos = ["BFS", "DFS", "A* (Manhattan)", "A* (Euclidiana)", "A* (Chebyshev)"]
        for algo in algos:
            tk.Radiobutton(control_panel, text=algo, variable=self.algoritmo_var, value=algo, bg="#f0f0f0", anchor="w").pack(fill=tk.X, padx=20)

        tk.Label(control_panel, text="AÇÕES", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(20, 5))
        tk.Button(control_panel, text="▶ EXECUTAR (Visual)", command=self.executar_visual, bg="#2ecc71", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(control_panel, text="📊 BENCHMARK (Mapa Atual)", command=self.executar_benchmark, bg="#3498db", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(control_panel, text="🧪 EXPERIMENTO COMPLETO", command=self.executar_experimento_global, bg="#9b59b6", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, padx=10, pady=5)
        tk.Button(control_panel, text="🔄 RESETAR", command=self.resetar_mapa, bg="#95a5a6", fg="white").pack(fill=tk.X, padx=10, pady=5)

        # Área de Resultados
        self.txt_result = tk.Text(control_panel, height=12, width=30, bg="white", font=("Consolas", 9))
        self.txt_result.pack(padx=5, pady=20, fill=tk.BOTH, expand=True)

        # Canvas do Labirinto
        self.canvas = tk.Canvas(main_frame, bg="white", highlightthickness=0)
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.carregar_dados_mapa()
        self.desenhar_grid()

    def carregar_dados_mapa(self):
        self._load_map_state(self.mapa_idx, self.pos_idx)

    def _load_map_state(self, m_idx, p_idx):
        raw = MAPAS[m_idx].strip()
        self.grid_chars = [list(row) for row in raw.split('\n')]
        self.rows = len(self.grid_chars)
        self.cols = len(self.grid_chars[0])
        
        sp, ep = CONFIGS_POSICOES[m_idx][p_idx]
        self.start = (min(sp[0], self.rows-2), min(sp[1], self.cols-2))
        self.end = (min(ep[0], self.rows-2), min(ep[1], self.cols-2))
        
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid_chars[r][c] in ('S', 'E'): self.grid_chars[r][c] = ' '
        
        if self.grid_chars[self.start[0]][self.start[1]] == '#': self.grid_chars[self.start[0]][self.start[1]] = ' '
        if self.grid_chars[self.end[0]][self.end[1]] == '#': self.grid_chars[self.end[0]][self.end[1]] = ' '

        self.lbl_mapa.config(text=f"Mapa {m_idx + 1}")
        self.lbl_pos.config(text=f"Posição {p_idx + 1}")

    def desenhar_grid(self):
        self.canvas.delete("all")
        self.rects = {}
        
        total_w = self.cols * TAM_CELULA
        total_h = self.rows * TAM_CELULA
        offset_x = (self.canvas.winfo_width() - total_w) // 2
        offset_y = (self.canvas.winfo_height() - total_h) // 2
        if offset_x < 0: offset_x = 10
        if offset_y < 0: offset_y = 10

        for r in range(self.rows):
            for c in range(self.cols):
                x1 = offset_x + c * TAM_CELULA
                y1 = offset_y + r * TAM_CELULA
                x2 = x1 + TAM_CELULA
                y2 = y1 + TAM_CELULA
                
                cor = CORES['LIVRE']
                if self.grid_chars[r][c] == '#': cor = CORES['PAREDE']
                if (r, c) == self.start: cor = CORES['START']
                if (r, c) == self.end: cor = CORES['END']
                
                tag = f"{r}_{c}"
                rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, fill=cor, outline="#bdc3c7", tags=tag)
                self.rects[(r, c)] = rect_id

    def atualizar_celula(self, pos, tipo):
        if pos == self.start or pos == self.end: return
        cor = CORES.get(tipo, CORES['LIVRE'])
        rect_id = self.rects.get(pos)
        if rect_id:
            self.canvas.itemconfig(rect_id, fill=cor)

    def mudar_mapa(self, delta):
        self.mapa_idx = (self.mapa_idx + delta) % len(MAPAS)
        self.carregar_dados_mapa()
        self.desenhar_grid()
        self.txt_result.delete(1.0, tk.END)

    def mudar_posicao(self, delta):
        self.pos_idx = (self.pos_idx + delta) % 5
        self.carregar_dados_mapa()
        self.desenhar_grid()
        self.txt_result.delete(1.0, tk.END)

    def resetar_mapa(self):
        self.desenhar_grid()
        self.txt_result.delete(1.0, tk.END)

    def get_neighbors(self, pos):
        r, c = pos
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] 
        neighbors = []
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.rows and 0 <= nc < self.cols and self.grid_chars[nr][nc] != '#':
                neighbors.append((nr, nc))
        return neighbors

    # --- Lógica de Busca ---
    def resolver(self, algoritmo, visual=False):
        start_time = time.perf_counter()
        came_from = {self.start: None}
        visitados = set()
        nodes_count = 0
        success = False

        # Heurísticas
        def h_manhattan(a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])
        def h_euclidean(a, b): return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)
        def h_chebyshev(a, b): return max(abs(a[0]-b[0]), abs(a[1]-b[1]))

        if algoritmo == "BFS":
            container = deque([self.start])
            visitados.add(self.start)
            pop_func = container.popleft
            push_func = container.append
        elif algoritmo == "DFS":
            container = [self.start]
            visitados.add(self.start)
            pop_func = container.pop
            push_func = container.append
        else: # A*
            container = [] 
            heapq.heappush(container, (0, self.start))
            g_score = {self.start: 0}
            if "Manhattan" in algoritmo: h = h_manhattan
            elif "Euclidiana" in algoritmo: h = h_euclidean
            else: h = h_chebyshev

        steps = 0
        while container:
            if visual:
                steps += 1
                if steps % 2 == 0: 
                    self.root.update()

            if "A*" in algoritmo:
                _, current = heapq.heappop(container)
                visitados.add(current)
            else:
                current = pop_func()
                if algoritmo == "DFS" and current not in visitados:
                    visitados.add(current)
            
            nodes_count += 1
            if visual: self.atualizar_celula(current, 'VISITADO')

            if current == self.end:
                success = True
                break
            
            for neighbor in self.get_neighbors(current):
                if "A*" in algoritmo:
                    new_g = g_score[current] + 1
                    if neighbor not in g_score or new_g < g_score[neighbor]:
                        g_score[neighbor] = new_g
                        f = new_g + h(neighbor, self.end)
                        heapq.heappush(container, (f, neighbor))
                        came_from[neighbor] = current
                        if visual: self.atualizar_celula(neighbor, 'FRONTEIRA')
                else:
                    if neighbor not in visitados and neighbor not in came_from:
                        if algoritmo == "BFS": 
                            visitados.add(neighbor)
                            push_func(neighbor)
                            came_from[neighbor] = current
                            if visual: self.atualizar_celula(neighbor, 'FRONTEIRA')
                        elif algoritmo == "DFS":
                            push_func(neighbor) 
                            came_from[neighbor] = current
                            if visual: self.atualizar_celula(neighbor, 'FRONTEIRA')

        path = []
        if success:
            curr = self.end
            while curr != self.start:
                path.append(curr)
                if visual: 
                    self.atualizar_celula(curr, 'CAMINHO')
                    self.root.update()
                curr = came_from[curr]
            path.append(self.start)
            path.reverse()
        
        end_time = time.perf_counter()
        duration_ms = (end_time - start_time) * 1000

        if visual:
            self.txt_result.delete(1.0, tk.END)
            self.txt_result.insert(tk.END, f"ALGO: {algoritmo}\nStatus: {'Sucesso' if success else 'Falha'}\nTempo: {duration_ms:.2f}ms\nNós: {nodes_count}\nCusto: {len(path)-1}\n")

        return {
            "name": algoritmo,
            "success": success,
            "cost": len(path)-1 if success else 0,
            "time": duration_ms,
            "nodes": nodes_count
        }

    def executar_visual(self):
        self.resetar_mapa()
        algo = self.algoritmo_var.get()
        self.resolver(algo, visual=True)

    def executar_benchmark(self):
        self.resetar_mapa()
        algos = ["BFS", "DFS", "A* (Manhattan)", "A* (Euclidiana)", "A* (Chebyshev)"]
        self.txt_result.delete(1.0, tk.END)
        self.txt_result.insert(tk.END, "RODANDO BENCHMARK...\n")
        self.root.update()
        
        resultados = []
        for algo in algos:
            res = self.resolver(algo, visual=False)
            resultados.append(res)
        
        self.txt_result.delete(1.0, tk.END)
        header = f"{'ALGO':<12} | {'NÓS':<5} | {'CUSTO':<5} | {'TEMPO'}\n"
        self.txt_result.insert(tk.END, header + "-"*45 + "\n")
        
        best_time = float('inf')
        winner = ""
        for res in resultados:
            nome = res['name'].replace("A* ", "").replace("Manhattan", "Manh").replace("Euclidiana", "Eucl").replace("Chebyshev", "Cheb")
            self.txt_result.insert(tk.END, f"{nome:<12} | {res['nodes']:<5} | {res['cost']:<5} | {res['time']:.2f}ms\n")
            if res['time'] < best_time and res['success']:
                best_time = res['time']
                winner = res['name']
        
        self.txt_result.insert(tk.END, "-"*45 + "\n")
        self.txt_result.insert(tk.END, f"🏆 Rápido: {winner}")

    # --- Experimento Completo ---
    def executar_experimento_global(self):
        # Janela de Progresso
        prog_win = tk.Toplevel(self.root)
        prog_win.title("Experimento: Critério Ótimo...")
        prog_win.geometry("400x150")
        lbl_status = tk.Label(prog_win, text="Calculando...")
        lbl_status.pack(pady=10)
        progress = ttk.Progressbar(prog_win, orient=tk.HORIZONTAL, length=300, mode='determinate')
        progress.pack(pady=10)
        
        algos = ["BFS", "DFS", "A* (Manhattan)", "A* (Euclidiana)", "A* (Chebyshev)"]
        total_steps = len(MAPAS) * 5 * len(algos)
        step_count = 0
        
        orig_map = self.mapa_idx
        orig_pos = self.pos_idx
        
        resultados_finais = []
        NUM_REPETICOES = 50

        for m_idx in range(len(MAPAS)):
            for p_idx in range(5):
                self._load_map_state(m_idx, p_idx)
                
                # Descobrir qual é o custo ótimoa através do BFS
                bfs_res = self.resolver("BFS", visual=False)
                custo_otimo = bfs_res['cost']
                
                best_algo_nodes = ("Nenhum", float('inf'))
                best_algo_time = ("Nenhum", float('inf'))
                
                cenario_res = []
                
                for algo in algos:
                    res = self.resolver(algo, visual=False)
                    
                    soma_tempo = 0
                    for _ in range(NUM_REPETICOES):
                        t0 = time.perf_counter()
                        self.resolver(algo, visual=False) 
                        soma_tempo += (time.perf_counter() - t0)
                    
                    media_tempo = (soma_tempo / NUM_REPETICOES) * 1000
                    res['time'] = media_tempo
                    cenario_res.append(res)
                    
                    if res['success'] and res['cost'] == custo_otimo:
                        # Melhor Time (entre os ótimos)
                        if media_tempo < best_algo_time[1]:
                            best_algo_time = (algo, media_tempo)
                        
                        # Menos Nós (entre os ótimos)
                        if res['nodes'] < best_algo_nodes[1]:
                            best_algo_nodes = (algo, res['nodes'])
                    
                    step_count += 1
                    progress['value'] = (step_count / total_steps) * 100
                    lbl_status.config(text=f"Mapa {m_idx+1} | Pos {p_idx+1} | {algo}")
                    prog_win.update()
                
                resultados_finais.append({
                    "mapa": m_idx + 1,
                    "posicao": p_idx + 1,
                    "custo_otimo": custo_otimo,
                    "vencedor_tempo": best_algo_time,
                    "vencedor_nos": best_algo_nodes,
                    "detalhes": cenario_res
                })

        prog_win.destroy()
        
        self.mapa_idx = orig_map
        self.pos_idx = orig_pos
        self.carregar_dados_mapa()
        self.desenhar_grid()
        
        self.mostrar_relatorio_final(resultados_finais)

    def mostrar_relatorio_final(self, dados):
        rel_win = tk.Toplevel(self.root)
        rel_win.title("Relatório Final (Critério: Custo Ótimo)")
        rel_win.geometry("1200x600")
        
        tk.Label(rel_win, text="RELATÓRIO: Vencedores consideram apenas quem achou o menor caminho possível", font=("Arial", 11, "bold")).pack(pady=10)
        
        frame_tabela = tk.Frame(rel_win)
        frame_tabela.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        cols = ("Mapa", "Pos", "Custo Ótimo", "Vencedor Tempo", "Tempo (ms)", "Vencedor Nós", "Nós", "Detalhes (Algo:Tempo)")
        tree = ttk.Treeview(frame_tabela, columns=cols, show='headings')
        
        # Configuração das colunas
        tree.heading("Mapa", text="Mapa")
        tree.column("Mapa", width=40, anchor="center")
        tree.heading("Pos", text="Pos")
        tree.column("Pos", width=40, anchor="center")
        tree.heading("Custo Ótimo", text="Custo")
        tree.column("Custo Ótimo", width=50, anchor="center")
        
        tree.heading("Vencedor Tempo", text="Melhor Tempo (Ótimo)")
        tree.column("Vencedor Tempo", width=140)
        tree.heading("Tempo (ms)", text="ms")
        tree.column("Tempo (ms)", width=70, anchor="center")
        
        tree.heading("Vencedor Nós", text="Menos Nós (Ótimo)")
        tree.column("Vencedor Nós", width=140)
        tree.heading("Nós", text="Qtd")
        tree.column("Nós", width=60, anchor="center")
        
        tree.heading("Detalhes (Algo:Tempo)", text="Comparativo Completo")
        tree.column("Detalhes (Algo:Tempo)", width=500)
        
        vsb = ttk.Scrollbar(frame_tabela, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=vsb.set)
        
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        
        for item in dados:
            detalhes_str = " | ".join([
                f"{r['name'].replace('A* ', 'A*').replace('Manhattan', 'M').replace('Euclidiana', 'E').replace('Chebyshev', 'C')}:{r['time']:.3f}ms" 
                for r in item['detalhes']
            ])
            
            tempo_val = f"{item['vencedor_tempo'][1]:.4f}" if item['vencedor_tempo'][1] != float('inf') else "-"
            nos_val = item['vencedor_nos'][1] if item['vencedor_nos'][1] != float('inf') else "-"
            
            tree.insert("", tk.END, values=(
                item['mapa'], 
                item['posicao'], 
                item['custo_otimo'],
                item['vencedor_tempo'][0], 
                tempo_val,
                item['vencedor_nos'][0],
                nos_val,
                detalhes_str
            ))

if __name__ == "__main__":
    root = tk.Tk()
    app = MazeApp(root)
    root.update()
    app.desenhar_grid()
    root.mainloop()