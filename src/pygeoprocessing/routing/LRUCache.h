#ifndef __LRUCACHE_H_INCLUDED__
#define __LRUCACHE_H_INCLUDED__

#include <list>
#include <map>
#include <assert.h>

using namespace std;

template <class KEY_T, class VAL_T,
    typename ListIter = typename list< typename pair<KEY_T,VAL_T> >::iterator,
    typename MapIter = typename map<KEY_T, ListIter>::iterator > class LRUCache{
private:
    list< pair<KEY_T,VAL_T> > item_list;
    map<KEY_T, ListIter> item_map;
    size_t cache_size;
private:
    void clean(list< typename pair<KEY_T, VAL_T> > &removed_value_list){
        while(item_map.size()>cache_size){
            ListIter last_it = item_list.end(); last_it --;
            MapIter map_it = item_map.find(last_it->first);
            removed_value_list.push_back(
                make_pair(last_it->first, last_it->second));
            item_map.erase(last_it->first);
            item_list.pop_back();
        }
    };
public:
    LRUCache(int cache_size_):cache_size(cache_size_){
            ;
    };

    ListIter begin() {
        return item_list.begin();
    }

    ListIter end() {
        return item_list.end();
    }

    void put(
            const KEY_T &key, const VAL_T &val,
            list< typedef pair<KEY_T, VAL_T> > &removed_value_list) {
        MapIter it = item_map.find(key);
        if(it != item_map.end()){
            item_list.erase(it->second);
            item_map.erase(it);
        }
        item_list.push_front(make_pair(key,val));
        item_map.insert(make_pair(key, item_list.begin()));
        return clean(removed_value_list);
    };
    bool exist(const KEY_T &key){
            return (item_map.count(key)>0);
    };
    VAL_T& get(const KEY_T &key){
            assert(exist(key));
            MapIter it = item_map.find(key);
            item_list.splice(item_list.begin(), item_list, it->second);
            return it->second->second;
    };
};
#endif
