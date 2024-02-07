from typing import List, Dict

class TreeNode:
    def __init__(self, value, id):
        self.value = value
        self.id = id
        self.children = []
    
class Tree:
    def __init__(self, root_value, root_id):
        self.root = TreeNode(root_value, root_id)
        self.nodes = {root_id: self.root}
    
    def add_child(self, parent_id, child_value, child_id):
        if parent_id in self.nodes:
            parent_node = self.nodes[parent_id]
            child_node = TreeNode(child_value, child_id)
            parent_node.children.append(child_node)
            self.nodes[child_id] = child_node
        else:
            print(f"Parent node with id {parent_id} not found")
    
    def find_by_id(self, id):
        return self.nodes.get(id, None)
    
    def print_tree(self, node, level=0):
        print("\t" * level + "->" + node.value)
        for child in node.children:
            self.print_tree(child, level + 1)   
             
