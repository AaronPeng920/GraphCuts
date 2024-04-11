from collections import deque
import warnings

class MaxFlow:
    def __init__(
        self, 
        node_num=0,             # node count without source node & sink node
        source=-2,              # source node idx
        sink=-1,                # sink node idx
        capacity_table=[],      # [{node_num: capacity}, {node_num: capacity}, ...]
        flow=0.0,               # flow 
        levels=[],
        segment=[],             # 
        can_reach_sink=[]
    ):
        self.node_num =  node_num
        self.source = source
        self.sink = sink
        self.capacity_table = capacity_table
        self.flow = flow
        self.levels = levels
        self.segment = segment
        self.can_reach_sink = can_reach_sink
    
    # ! You can only call one #
    def set_nodes(self, node_num):
        self.node_num = node_num + 2 
        self.source = self.node_num - 2
        self.sink = self.node_num - 1
        self.capacity = [{} for i in range(self.node_num)]
    
    def add_edge(self, node1, node2, f_capacity, b_capacity):
        """add eage

        Args:
            node1 (int): node1 idx
            node2 (int): node2 idx
            f_capacity (float): forward(node1 -> node2) capacity
            b_capacity (float): backward(node2 -> node2) capacity
        """
        if node1 < 0 or node1 >= self.node_num - 2 or node2 < 0 or node2 >= self.node_num - 2:
            raise Exception('add_egde: node index is out of scope.')
        self.capacity[node1][node2] = f_capacity
        self.capacity[node2][node1] = b_capacity
    
    def add_tedge(self, node, s_capacity, t_capacity):
        """add teage

        Args:
            node (int): node idx
            s_capacity (float): source -> node capacity
            t_capacity (float): node -> sink capacity
        """
        if node < 0 or node >= self.node_num - 2:
            raise Exception('add_tedge: node index is out of scope.')
        if s_capacity > 0:
            self.capacity[self.source][node] = s_capacity
        if t_capacity > 0:
            self.capacity[node][self.sink] = t_capacity
            
    def bfs(self):
        """Breath First Search
        
        Aegs:
            return: sink node level, 1 for source
        """
        self.levels = [0] * self.node_num           # layer level for each node, 1 for source, 0 for not visited
        queue = deque()
        queue.append(self.source)
        self.levels[self.source] = 1        

        level_cnt = {}                  # node count for each layer
        level_cnt[1] = 1
        reach_sink_cnt = 0              # node count which can reach sink in bfs method
        father = [-1] * self.node_num               # father node idx for each node, -1 for init/no father
        self.can_reach_sink = [False] * self.node_num       # can reach sink for each node
        self.can_reach_sink[self.sink] = True

        break_level = -1            # sink node level
        while queue:
            cur = queue.popleft()
            if break_level != -1 and self.levels[cur] == break_level:
                break
            for nei, cap in self.capacity[cur].items():     # (next_node_idx, edge capacity)
                if cap > 0 and nei == self.sink:
                    break_level = self.levels[cur] + 1
                    reach_sink_cnt += 1
                    tmp = cur
                    while(tmp != -1):
                        self.can_reach_sink[tmp] = True
                        tmp = father[tmp]
                if self.levels[nei] == 0 and cap > 0:
                    father[nei] = cur
                    self.levels[nei] = self.levels[cur] + 1

                    if self.levels[nei] not in level_cnt:
                        level_cnt[self.levels[nei]] = 0
                    level_cnt[self.levels[nei]] += 1

                    queue.append(nei)
                    
        if self.levels[self.sink] > 0:
            print("------------------ BFS Start ------------------")
            print("Cur flow: ", self.flow)
            print("Bfs: sink node is in level", self.levels[self.sink])
            print("Bfs: number of paths that reaches sink: ", reach_sink_cnt)
            print("------------------ BFS End ------------------")

        return self.levels[self.sink]
    
    def dfs(self, cur, cur_max_inbound, path, sink_level):
        """Depth First Search

        Args:
            cur (int): current node idx
            cur_max_inbound (float): max inbound for current node
            path (list): DFS search order
            sink_level (int): sink node level

        Returns:
            _type_: _description_
        """
        if cur_max_inbound <= 0:
            return 0
        if cur == self.sink:
            return cur_max_inbound
        cur_outbound = 0
        for nei, cap in self.capacity[cur].items():
            if self.levels[nei] == self.levels[cur] + 1 and self.levels[nei] <= sink_level and cap > 0:
                flow = self.dfs(nei, min(cur_max_inbound - cur_outbound, cap), path + [nei], sink_level)
                self.capacity[cur][nei] -= flow
                if nei != self.sink and cur != self.source:
                    self.capacity[nei][cur] += flow
                cur_outbound += flow
        
        return cur_outbound
    
    def maxflow(self):
        """Dicnic maxflow algorithm

        Returns:
            float: maxflow 
        """
        self.flow = 0
        while(True):
            sink_level = self.bfs()
            if sink_level == 0:
                break
            path = []
            flow = self.dfs(self.source, float('inf'), path, sink_level)
            self.flow = self.flow + flow

        return self.flow

    def get_segment(self, node_index):
        """get which segment for node

        Args:
            node_index (int): node idx

        Returns:
            0/1: foreground/background
        """
        if node_index < 0 or node_index >= self.node_num - 2:
            raise Exception('get_segment: node index is out of scope.')
        if len(self.segment) == 0:
            self.set_object_nodes()

        return self.segment[node_index]

    def set_object_nodes(self):
        """split foreground and background"""
        self.segment = self.node_num * [1]
        queue = deque()
        queue.append(self.source)
        while queue:
            cur = queue.popleft()
            for nei, cap in self.capacity[cur].items():
                if self.segment[nei] == 1 and cap == 0:
                    queue.append(nei)
                    self.segment[nei] = 0

    